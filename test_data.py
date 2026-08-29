# test_data.py
import json
from augmentations import get_train_augmentation

print("🔍 Testing Person A's data with my augmentations...")

# Load the splits using the correct keys
with open('train_splits.json', 'r') as f:
    data = json.load(f)

train_paths = data["train_paths"]
train_labels = data["train_labels"]

print(f"   Found {len(train_paths)} training images")

# Now test with Person A's dataset class
try:
    from dataset import ImageDataset
    print("   ✅ ImageDataset found in dataset.py")
    
    # Test with first 5 images
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
    
except ImportError as e:
    print(f"   ❌ Could not import ImageDataset: {e}")
    print("   Please make sure dataset.py has the ImageDataset class")