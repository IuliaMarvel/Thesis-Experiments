import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import pickle
from torchvision import datasets, transforms
from utils import clamp

NUM_CLASSES = 100 

data_mean = (0.5071, 0.4865, 0.4409)
data_std = (0.2673, 0.2564, 0.2762)

loss_weight = 0.3
fgsm_step = 1
delta_init = 'random'
criterion = nn.CrossEntropyLoss()


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit,device, args=None):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    scaler = torch.cuda.amp.GradScaler()
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            with torch.cuda.amp.autocast():
                output = model(X + delta)
                index = torch.where(output.max(1)[1] == y)[0]
                if len(index) == 0:
                    break
                loss = F.cross_entropy(output, y)
            scaler.scale(loss).backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index, :, :, :], upper_limit - X[index, :, :, :])
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts, epsilon, alpha, lower_limit, upper_limit,device, args=None):

    examples_per_class = {i : 0 for i in range(NUM_CLASSES)}
    correct_per_class = {i : 0 for i in range(NUM_CLASSES)}

    pgd_loss, pgd_acc = 0, 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit,device, args=args)
        with torch.no_grad():
            output = model(X + pgd_delta)
            preds = output.max(1)[1]
            for i in range(NUM_CLASSES):
                examples_per_class[i] += len(y[y==i])
                correct_per_class[i] += (preds[y==i] == i).sum().item()
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        acc_per_class = {}
    for i in range(NUM_CLASSES):
        acc_per_class[i] = correct_per_class[i] / examples_per_class[i]
    return pgd_loss/n, pgd_acc/n,acc_per_class


def evaluate_standard(test_loader, model,device, args=None):
    
    examples_per_class = {i : 0 for i in range(NUM_CLASSES)}
    correct_per_class = {i : 0 for i in range(NUM_CLASSES)}
    
    test_loss, test_acc = 0, 0
    n = 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            preds = output.max(1)[1]
            for i in range(NUM_CLASSES):
                examples_per_class[i] += len(y[y==i])
                correct_per_class[i] += (preds[y==i] == i).sum().item()
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    acc_per_class = {}
    for i in range(NUM_CLASSES):
        acc_per_class[i] = correct_per_class[i] / examples_per_class[i]
    return test_loss/n, test_acc/n,acc_per_class

def train_val_cycle(model,train_loader,test_loader,optimizer,scaler,epochs,device,delta_init,epsilon,alpha):
    loop = tqdm(range(epochs))
    acc_history = {'train': [], 'standart_val': [], 'attack_val': []}
    
    mu = torch.tensor(data_mean).view(3, 1, 1).to(device)
    std = torch.tensor(data_std).view(3, 1, 1).to(device)
    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)
    epsilon = (epsilon / 255.) / std
    alpha = (alpha / 255.) / std
    
    for epoch in loop:
        print('epoch started')
        train_loss,train_acc,train_n = 0,0,0
        # if epoch > 12:
        # optimizer.param_groups[0]['lr'] = 1e-4
        model.train()
        print('training...')
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            if delta_init != 'previous':
                delta = torch.zeros_like(X).to(device)
            if delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            if fgsm_step == 1:
                delta.requires_grad = True
                for _ in range(1):
                    output = model(X + delta[:X.size(0)])
                    loss = F.cross_entropy(output, y)
                    scaler.scale(loss).backward()
                    grad = delta.grad.detach()
                    scaler.step(optimizer)
                    scaler.update()
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                delta = delta.detach()
                output = model(X + delta[:X.size(0)])
                loss = (1 - loss_weight) * criterion(output, y) + loss_weight * criterion(model(X), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        
        train_acc = train_acc / train_n
        acc_history['train'].append(train_acc)
        model.eval()
        print('Evaluate on attacs')
        pgd_loss, pgd_acc, acc_cl_pgd_at = evaluate_pgd(test_loader, model, 1, 5, epsilon, alpha, 
                                    lower_limit, upper_limit,device)
        acc_history['attack_val'].append(pgd_acc)
        
        st_loss, st_acc, acc_cl_st_at = evaluate_standard(test_loader, model,device)
        acc_history['standart_val'].append(st_acc)
        print('Evaluate on clear')
        loop.set_description(f'train acc: {round(train_acc,3)} test st_acc: {st_acc} test at_acc: {pgd_acc}')

    return acc_history

