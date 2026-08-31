# run_robustness_table.py
# Person D runs this on Sunday to generate the full results table

import torch
from torch.utils.data import DataLoader
import json
import os
from transforms.transforms import get_test_conditions
from PIL import Image
import numpy as np

# =============================================
# Fill PERSON A'S CODE 
# =============================================

class ImageDataset:
    """Simplified version - Person A will replace with their actual class"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_np = np.array(img)
        if self.transform:
            img_np = self.transform(image=img_np)['image']
        return img_np, self.labels[idx]


# =============================================
# Fill PERSON D'S evaluate.py
# =============================================

def evaluate(model, val_loader):
    """Person D's evaluation function - simplified version here"""
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            # Move to GPU if available
            if torch.cuda.is_available():
                images = images.cuda()
            
            # Get predictions
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Handle different output shapes
            if len(probs.shape) > 1:
                probs = probs.flatten()
            
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, preds_binary)
    
    return acc, auc


# =============================================
# MAIN FUNCTION
# =============================================

def run_robustness_table(model_paths, val_paths, val_labels, model_loader_fn):
    """
    Run robustness evaluation across all test conditions.
    
    Args:
        model_paths: dict like {"plain": "path.pt", "augmented": "path.pt"}
        val_paths: list of image file paths (from Person A)
        val_labels: list of labels (from Person A)
        model_loader_fn: function that loads a model from a checkpoint
    
    Returns:
        dict: results table
    """
    test_conditions = get_test_conditions()
    results = {}
    
    for model_name, checkpoint_path in model_paths.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")
        
        # Load the model
        model = model_loader_fn(checkpoint_path)
        
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        
        results[model_name] = {}
        
        for condition_name, transform in test_conditions.items():
            print(f"\n  Testing: {condition_name}")
            
            # Create dataset with this transform
            dataset = ImageDataset(val_paths, val_labels, transform=transform)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # Evaluate
            acc, auc = evaluate(model, loader)
            
            # Store results
            results[model_name][condition_name] = {
                "accuracy": round(acc * 100, 2),  # as percentage
                "auc": round(auc, 4)
            }
            
            print(f"    Accuracy: {acc*100:.2f}%")
            print(f"    AUC: {auc:.4f}")
    
    return results


# =============================================
# EXAMPLE OF HOW TO USE IT
# =============================================

if __name__ == "__main__":
    """
    PERSON D - INSTRUCTIONS:
    1. Fill in the paths below
    2. Define model_loader_fn to load your model
    3. Get val_paths and val_labels from Person A
    4. Run this script
    """
    
    # ===== FILL THESE IN =====
    # Fill from person B
    MODEL_PATHS = {
        "plain": "models/plain_model.pt",        # Person B's plain model
        "augmented": "models/augmented_model.pt" # Person B's augmented model
    }
    
    # Fill values from Person A
    VAL_PATHS = []      # e.g., glob.glob("data/cifake/val/*/*.jpg")
    VAL_LABELS = []     # e.g., [0, 1, 0, 1, ...]
    
    # Fill the function from Person B
    def load_model(checkpoint_path):
       
        pass
    
    # ===== RUN THE TABLE =====
    results = run_robustness_table(
        model_paths=MODEL_PATHS,
        val_paths=VAL_PATHS,
        val_labels=VAL_LABELS,
        model_loader_fn=load_model
    )
    
    # ===== SAVE RESULTS =====
    with open("robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("RESULTS SAVED TO: robustness_results.json")
    print("="*50)
    
    # Print table
    print("\n📊 RESULTS TABLE:")
    print("-" * 60)
    print(f"{'Condition':<15} {'Plain Acc':<12} {'Plain AUC':<12} {'Aug Acc':<12} {'Aug AUC':<12}")
    print("-" * 60)
    
    for condition in results["plain"].keys():
        p_acc = results["plain"][condition]["accuracy"]
        p_auc = results["plain"][condition]["auc"]
        a_acc = results["augmented"][condition]["accuracy"]
        a_auc = results["augmented"][condition]["auc"]
        print(f"{condition:<15} {p_acc}%{'':<8} {p_auc:<12} {a_acc}%{'':<8} {a_auc:<12}")