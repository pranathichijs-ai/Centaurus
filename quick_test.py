from augmentations import get_train_augmentation, get_test_conditions
from PIL import Image
import numpy as np
import os

print("Looking for test images...")

# Check for test images
test_image = None
possible_paths = ["test_real.jpg", "test_fake.jpg", "test.jpg"]

for path in possible_paths:
    if os.path.exists(path):
        test_image = path
        break

# If no test images, look in train folder
if test_image is None:
    for folder in ["train/REAL", "train/FAKE"]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    test_image = os.path.join(folder, f)
                    break
            if test_image:
                break

if test_image is None:
    print("No test image found!")
    print("Please copy an image from train/REAL/ or train/FAKE/ to test")
    exit()

print(f"Found test image: {test_image}")

# Load and test
img = Image.open(test_image).convert("RGB")
img_np = np.array(img)
print(f"   Image shape: {img_np.shape}")

# Test training augmentation
print("\n Testing training augmentation...")
try:
    train_aug = get_train_augmentation()
    result = train_aug(image=img_np)['image']
    print(f"   Train augmentation works! Output shape: {result.shape}")
except Exception as e:
    print(f"   Error: {e}")

# Test all test conditions
print("\n Testing test conditions...")
test_conditions = get_test_conditions()
for name, transform in test_conditions.items():
    try:
        result = transform(image=img_np)['image']
        print(f"   {name}: output shape {result.shape}")
    except Exception as e:
        print(f"   {name}: Error - {e}")

print("\n All tests complete!")