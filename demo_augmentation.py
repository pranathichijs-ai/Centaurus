# demo_augmentation.py
# Show augmentation working on a sample image (for video)

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from augmentations import get_train_augmentation, get_test_conditions

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
    plt.show()
    print(f"Demo saved to {output_path}")

if __name__ == "__main__":
    # Use any image you have
    demo_augmentation("sample_images/test_0.jpg")