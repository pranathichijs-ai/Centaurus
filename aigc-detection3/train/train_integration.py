# train_integration.py
# Person C - Augmentation integration into training loop
# Using small dataset as placeholder

import torch
from torch.utils.data import DataLoader
from transforms.transforms import get_train_augmentation
from dataset import ImageDataset
import json

print("=" * 60)
print("🔄 Person C - Augmentation Pipeline Integration")
print("=" * 60)

# Step 1: Load small dataset (placeholder)
print("\n📦 Step 1: Loading small dataset...")

# Load the splits
with open('train_splits.json', 'r') as f:
    data = json.load(f)

train_paths = data["train_paths"]
train_labels = data["train_labels"]

# Use only 20 images as placeholder
placeholder_size = 20
small_paths = train_paths[:placeholder_size]
small_labels = train_labels[:placeholder_size]

print(f"   ✅ Loaded {len(small_paths)} images as placeholder")

# Step 2: Create dataset with YOUR augmentation
print("\n🔄 Step 2: Creating dataset with augmentation...")

train_dataset = ImageDataset(
    small_paths,
    small_labels,
    transform=get_train_augmentation()  # 👈 YOUR AUGMENTATION
)

print(f"   ✅ Dataset created with augmentation")
print(f"   Dataset size: {len(train_dataset)}")

# Step 3: Create DataLoader
print("\n📊 Step 3: Creating DataLoader...")

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

print(f"   ✅ DataLoader created")
print(f"   Batch size: 4")
print(f"   Number of batches: {len(train_loader)}")

# Step 4: Test one batch
print("\n🧪 Step 4: Testing one batch...")

for images, labels in train_loader:
    print(f"   ✅ Batch loaded successfully!")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Labels: {labels.tolist()}")
    break

print("\n" + "=" * 60)
print("✅ Augmentation pipeline integrated successfully!")
print("   Ready for Person B's training loop")
print("   DataLoader can be passed to model for training")
print("=" * 60)

# Step 5: Show how Person B will use it
print("\n📝 For Person B - How to use this:")
print("""
from train_integration import train_loader

# In your training loop:
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
""")