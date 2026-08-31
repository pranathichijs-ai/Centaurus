

# Example: how to load the training split
# import json
# with open("train_splits.json") as f:
# data = json.load(f)
# train_paths = data["train_paths"]
# train_labels = data["train_labels"]
# dataset.py
# Person A's dataset class - using Albumentations

from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ImageDataset(Dataset):
    """
    Custom Dataset class for loading images.
    Compatible with Albumentations transforms.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_np = np.array(img)
        
        if self.transform:
            # Albumentations expects numpy array
            transformed = self.transform(image=img_np)
            img = transformed['image']
        else:
            # Fallback: convert to tensor manually
            import torch
            img = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0
        
        return img, self.labels[idx]