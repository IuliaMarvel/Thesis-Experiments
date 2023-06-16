import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from torchvision import datasets, transforms

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_transforms(augments=[True, False]):
    data_mean = (0.5071, 0.4865, 0.4409)
    data_std = (0.2673, 0.2564, 0.2762)
    # types of transform: with augmentation and without
    transform_augmented = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)])
    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)])
    # choose transform according func params
    train_transform = transform_augmented if augments[0] else transform_clean
    test_transform = transform_augmented if augments[1] else transform_clean

    return train_transform, test_transform

def get_loaders(data_dir='cifar-100', batch_size=32, augments=[True, False], shuffles=[True, False]):
    train_transform, test_transform = prepare_transforms(augments)
    train_dataset = datasets.CIFAR100(
            data_dir, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(
            data_dir, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size, shuffle=shuffles[0],
        pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size, shuffle=shuffles[1],
        pin_memory=True, num_workers=2)

    return train_loader, test_loader

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_label(c):
  labels = []
  if c in hard_classes:
    labels.append('hard')
  elif c in easy_classes:
    labels.append('easy')
  elif c in random_classes:
    labels.append('random')
  else:
    labels.append('no interes')
  return labels

def get_loader(data_dir, X, y, batch_size,train=True, augment=True, shuffle=True, args=None):

    train_transform, test_transform = prepare_transforms([augment, False])
    transform = train_transform if train else test_transform
    dataset = datasets.CIFAR100(
            data_dir, train=train, transform=transform, download=True
        ) # we don't care what data here, we made it for getting appropriate transform
    dataset.data = np.concatenate(X,axis=0)
    dataset.targets = list(torch.cat(y,dim=0))

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, num_workers=2,
    )
    return loader