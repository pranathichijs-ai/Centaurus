# notebooks/test_data.py
import json
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now imports should work from anywhere
from transforms.transforms import get_train_augmentation
from data.dataset import ImageDataset

print("🔍 Testing Person A's data with my augmentations...")

# Load the data
with open(os.path.join(project_root, 'data/train_splits.json'), 'r') as f:
    data = json.load(f)

train_paths = data["train_paths"]
train_labels = data["train_labels"]

print(f"   Found {len(train_paths)} training images")

# Test with 5 images
dataset = ImageDataset(
    train_paths[:5], 
    train_labels[:5], 
    transform=get_train_augmentation()
)

print(f"✅ Dataset works! Size: {len(dataset)}")

# Test getting one sample
img, label = dataset[0]
print(f"   Image shape: {img.shape}")
print(f"   Label: {label}")
print("✅ All good!")