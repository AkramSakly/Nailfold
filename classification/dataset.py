"""
dataset.py
----------
PyTorch Dataset for NFC images + the paper's augmentation pipeline.

Augmentations (paper, "Data augmentation"):
  - random rotation up to 20% of a full rotation  (0.20 * 360 = 72 degrees)
  - random horizontal flip
  - random vertical flip
  - brightness and contrast jitter

Preprocessing:
  - resize to 224x224 (EfficientNet-B0 input, paper Table 3 Stage 0)
  - normalize with ImageNet mean/std (since backbone is ImageNet-pretrained)
"""

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=72),          # 20% of 360
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class NFCDataset(Dataset):
    """rows: list of dicts with keys 'path' and 'label'."""

    def __init__(self, rows, train: bool):
        self.rows = rows
        self.tf = build_transforms(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")   # handles RGBA inputs
        x = self.tf(img)
        y = torch.tensor(int(r["label"]), dtype=torch.long)
        return x, y