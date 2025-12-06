# data/fashionmnist.py

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_fashion_loaders(batch_size, root: str = "./data"):
    # Training: με augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    
    # Test: χωρίς augmentation
    test_transform = transforms.ToTensor()

    train_set = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=train_transform,  # <-- εδώ
    )

    test_set = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=test_transform,  # <-- εδώ
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,      # <-- bonus: πιο γρήγορο
        pin_memory=True,    # <-- bonus
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader