"""
Builds PyTorch DataLoaders from the processed train/val/test ImageFolder
directories created by scripts/prepare_data.py.
"""

import json
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(processed_dir, image_size=224, batch_size=32, num_workers=4):
    """
    Returns (train_loader, val_loader, test_loader, class_names).
    Also writes configs/classes.json mapping index -> class name,
    which detect.py relies on at inference time.
    """
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")
    test_dir = os.path.join(processed_dir, "test")

    for d in (train_dir, val_dir, test_dir):
        if not os.path.isdir(d) or not os.listdir(d):
            raise FileNotFoundError(
                f"Expected populated directory at '{d}'. "
                f"Did you run scripts/prepare_data.py first?"
            )

    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(image_size, train=True))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(image_size, train=False))
    test_ds = datasets.ImageFolder(test_dir, transform=build_transforms(image_size, train=False))

    class_names = train_ds.classes
    os.makedirs("configs", exist_ok=True)
    with open("configs/classes.json", "w") as f:
        json.dump({str(i): name for i, name in enumerate(class_names)}, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader, class_names
