import os
import random
import numpy as np 
from PIL import Image
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive, extract_archive
from torchvision.datasets import CIFAR10, CIFAR100, SVHN


CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
general_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def get_svhn_test_loader(path="/ssd1/chenwy", batch_size=64, num_workers=2):
    test_data = SVHN(root=path, split='test', download=True, transform=general_test_transform)
    test_queue_svhn = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return test_queue_svhn


def get_cifar_test_loader(path="/ssd1/chenwy", batch_size=64, num_workers=2):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_data = datasets.CIFAR10(root=path, train=False, download=True, transform=test_transform)
    test_queue_cifar10 = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return test_queue_cifar10


def svhn_dataloaders(batch_size=128, data_dir='/ssd1/chenwy', num_workers=2):

    train_transform = transforms.Compose([
        transforms.RandomCrop([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD) # TODO SVHN's mean/std
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    print('Dataset information: SVHN\t 90% of 73257 images for training \t 10%% images for validation\t')
    print('Data augmentation = randomcrop(32) + randomhorizontalflip')

    indice = list(range(73257))
    random.shuffle(indice)
    train_set = Subset(SVHN(root=data_dir, split='train', transform=train_transform, download=True), indice[:int(73257*0.9)])
    val_set = Subset(SVHN(root=data_dir, split='train', transform=test_transform, download=True), indice[int(73257*0.9):])
    test_set = SVHN(root=data_dir, split='test', transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar10_dataloaders(batch_size=128, data_dir='/ssd1/chenwy', num_workers=2):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    print('Dataset information: CIFAR-10\t 45000 images for training \t 500 images for validation\t')
    print('10000 images for testing')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def cifar100_dataloaders(batch_size=128, data_dir='/ssd1/chenwy', num_workers=2):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t')
    print('10000 images for testing')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar_c_testloader(corruption, level=-1, data_dir='/ssd1/chenwy', num_classes=10, test_batch_size=100, num_workers=2):
    '''
    Returns:
        test_c_loader: corrupted testing set loader (original cifar10-C)
    CIFAR10-C has 50,000 test images. 
    The first 10,000 images in each .npy are of level 1 severity, and the last 10,000 are of level 5 severity.
    level: 0~4 indicates corruption strength, -1 indicates all
    '''

    # # download:
    # url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
    # root_dir = data_dir
    # tgz_md5 = '56bf5dcef84df0e2308c6dcbcbbd8499'
    # if not os.path.exists(os.path.join(root_dir, 'CIFAR-10-C.tar')):
    #     download_and_extract_archive(url, root_dir, extract_root=root_dir, md5=tgz_md5)
    # elif not os.path.exists(os.path.join(root_dir, 'CIFAR-10-C')):
    #     extract_archive(os.path.join(root_dir, 'CIFAR-10-C.tar'), to_path=root_dir)

    if num_classes==10:
        CIFAR = datasets.CIFAR10
        base_c_path = os.path.join(data_dir, 'CIFAR-10-C')
    elif num_classes==100:
        CIFAR = datasets.CIFAR100
        base_c_path = os.path.join(data_dir, 'CIFAR-100-C')
    else:
        raise Exception('Wrong num_classes %d' % num_classes)
    
    # test set:
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_set = CIFAR(data_dir, train=False, transform=test_transform, download=False)
    
    # replace clean data with corrupted data:
    if level == -1:
        test_set.data = np.load(os.path.join(base_c_path, '%s.npy' % corruption))
        test_set.targets = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))
    else:
        assert isinstance(level, int) and level >= 0 and level <= 4
        test_set.data = np.load(os.path.join(base_c_path, '%s.npy' % corruption))[level*10000: (level+1)*10000]
        test_set.targets = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))[level*10000: (level+1)*10000]
    print('loader for %s ready level %d' % (corruption, level))

    test_c_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_c_loader