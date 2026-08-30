# notebooks/demo_augmentation.py
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transforms.transforms import get_train_augmentation, get_test_conditions

def demo_augmentation(image_path, output_path="demo_output.png"):
    """Show original + all augmentations side by side"""
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    
    # Get all transforms
    train_aug = get_train_augmentation()
    test_conditions = get_test_conditions()
    
    # Apply each
    results = {"Original": img_np}
    
    # Show 3 examples of training augmentation (random each time)
    for i in range(3):
        aug_result = train_aug(image=img_np)['image']
        # Convert from tensor to numpy for display
        aug_img = aug_result.permute(1, 2, 0).numpy()
        # Denormalize (roughly)
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        aug_img = aug_img * std + mean
        aug_img = np.clip(aug_img, 0, 1)
        results[f"Train Aug {i+1}"] = aug_img
    
    # Show each test condition
    for name, transform in test_conditions.items():
        result = transform(image=img_np)['image']
        img_display = result.permute(1, 2, 0).numpy()
        img_display = img_display * std + mean
        img_display = np.clip(img_display, 0, 1)
        results[name] = img_display
    
    # Create grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (name, img_display) in enumerate(results.items()):
        if i >= len(axes):
            break
        axes[i].imshow(img_display)
        axes[i].set_title(name, fontsize=12)
        axes[i].axis('off')
    
    # Hide unused subplots
    for j in range(len(results), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Demo image saved to {output_path}")
    plt.show()


def find_test_image():
    """Find any image in the project"""
    locations = [
        "sample_images",
        "data/cifake/train/REAL",
        "data/cifake/train/FAKE",
        "."
    ]
    
    for location in locations:
        full_path = os.path.join(project_root, location)
        if os.path.exists(full_path):
            for f in os.listdir(full_path):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(full_path, f)
    return None

if __name__ == "__main__":
    # Try to find any test image
    test_image = None
    
    # Check if user provided an image
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = find_test_image()
    
    if test_image is None:
        print("❌ No test image found!")
        print("Usage: python notebooks/demo_augmentation.py [path_to_image]")
        print("Or copy an image to sample_images/")
        exit()
    
    print(f"📷 Using image: {test_image}")
    demo_augmentation(test_image)