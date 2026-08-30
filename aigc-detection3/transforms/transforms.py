# transforms/transforms.py
# Person C - Complete augmentation pipeline

import albumentations as A
from albumentations.pytorch import ToTensorV2

# These are the exact CLIP normalization values - DO NOT CHANGE
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def get_train_augmentation():
    """
    Training-time augmentation pipeline.
    Applies ONE random distortion 80% of the time.
    Returns: Albumentations Compose object
    """
    return A.Compose([
        # Always resize to what CLIP expects
        A.Resize(224, 224),
        
        # Pick exactly ONE distortion (not all at once)
        A.OneOf([
            # JPEG compression - using quality_lower/upper (older API)
            A.ImageCompression(quality_lower=30, quality_upper=90, p=1.0),
            
            # Gaussian blur
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=1.0),
            
            # Gaussian noise
            A.GaussNoise(var_limit=(2.0, 10.0), p=1.0),
            
            # Color jitter
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=1.0),
            
            # Random crop - using height/width (older API)
            A.RandomResizedCrop(height=224, width=224, scale=(0.64, 1.0), p=1.0),
        ], p=0.8),  # 80% chance of applying distortion
        
        # CLIP's exact normalization
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2(),
    ])


def get_test_conditions():
    """
    Fixed test conditions for robustness evaluation.
    Each condition uses a FIXED setting (not random).
    Returns: dict of condition_name -> Albumentations Compose
    """
    
    # Base transforms that EVERY condition shares
    base_transforms = [
        A.Resize(224, 224),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2(),
    ]
    
    return {
        "Clean": A.Compose(base_transforms.copy()),
        
        "JPEG_q30": A.Compose([
            A.Resize(224, 224),
            A.ImageCompression(quality_lower=30, quality_upper=30, p=1.0),
            A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ToTensorV2(),
        ]),
        
        "Blur_sigma_2": A.Compose([
            A.Resize(224, 224),
            A.GaussianBlur(sigma_limit=(2.0, 2.0), p=1.0),
            A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ToTensorV2(),
        ]),
        
        "Crop_80": A.Compose([
            A.Resize(224, 224),
            A.RandomResizedCrop(height=224, width=224, scale=(0.8, 0.8), p=1.0),
            A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ToTensorV2(),
        ]),
    }


if __name__ == "__main__":
    print("✅ transforms.py loaded successfully!")
    print("   get_train_augmentation() ready")
    print("   get_test_conditions() ready")
    conditions = get_test_conditions()
    print(f"   {len(conditions)} test conditions:")
    for name in conditions.keys():
        print(f"      - {name}")