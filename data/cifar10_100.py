# data/cifar10_100.py

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(batch_size, root: str = "./data"):
    """
    CIFAR-10 loaders
    - Training: data augmentation (crop + flip)
    - Testing: no augmentation
    - NO normalization (values stay in [0,1] for rate encoding)
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),   # -> [0,1]
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),   # -> [0,1]
    ])

    train_set = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform_train,
    )

    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_cifar100_loaders(batch_size, root: str = "./data"):
    """
    CIFAR-100 loaders
    - Training: data augmentation (crop + flip)
    - Testing: no augmentation
    - NO normalization (values stay in [0,1] for rate encoding)
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),   # -> [0,1]
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),   # -> [0,1]
    ])

    train_set = datasets.CIFAR100(
        root=root,
        train=True,
        download=True,
        transform=transform_train,
    )

    test_set = datasets.CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader
