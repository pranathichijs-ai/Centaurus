# eval/robustness_eval.py
# Person C - Runs robustness evaluation (including WildFake)

import torch
from torch.utils.data import DataLoader
import json
import os
from transforms.transforms import get_test_conditions
from data.dataset import ImageDataset
from eval.evaluate import evaluate

# =============================================
# MAIN FUNCTION
# =============================================

def run_robustness_table(model_paths, val_paths, val_labels, wildfake_paths, wildfake_labels, model_loader_fn):
    """
    Run robustness evaluation across all test conditions.
    
    Args:
        model_paths: dict like {"plain": "path.pt", "augmented": "path.pt"}
        val_paths: list of image file paths (from Person A)
        val_labels: list of labels (from Person A)
        wildfake_paths: list of WildFake image paths (from Person A)
        wildfake_labels: list of WildFake labels (from Person A)
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
        
        # Run on all test conditions (Clean, JPEG_q30, Blur, Crop)
        for condition_name, transform in test_conditions.items():
            print(f"\n  Testing: {condition_name}")
            
            # Create dataset with this transform
            dataset = ImageDataset(val_paths, val_labels, transform=transform)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # Evaluate
            acc, auc = evaluate(model, loader)
            
            # Store results
            results[model_name][condition_name] = {
                "accuracy": round(acc * 100, 2),
                "auc": round(auc, 4)
            }
            
            print(f"    Accuracy: {acc*100:.2f}%")
            print(f"    AUC: {auc:.4f}")
        
        # =============================================
        # 🆕 NEW: Run on WildFake (unseen generator)
        # =============================================
        print(f"\n  Testing: WildFake (unseen generator)")
        
        # Use clean transform for WildFake (no distortion)
        clean_transform = test_conditions["Clean"]
        dataset = ImageDataset(wildfake_paths, wildfake_labels, transform=clean_transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        acc, auc = evaluate(model, loader)
        
        results[model_name]["WildFake"] = {
            "accuracy": round(acc * 100, 2),
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
    PERSON C - INSTRUCTIONS:
    1. Fill in the paths below
    2. Define model_loader_fn to load your model (Person B provides this)
    3. Get val_paths, val_labels, wildfake_paths, wildfake_labels from Person A
    4. Run this script
    """
    
    # ===== LOAD DATA FROM PERSON A =====
    print("\n📦 Loading data from Person A...")
    
    # Load validation data
    with open('val_splits.json', 'r') as f:
        val_data = json.load(f)
    val_paths = val_data["val_paths"]
    val_labels = val_data["val_labels"]
    print(f"   ✅ Validation: {len(val_paths)} images")
    
    # Load WildFake data
    with open('wildfake_splits.json', 'r') as f:
        wildfake_data = json.load(f)
    wildfake_paths = wildfake_data["wildfake_paths"]
    wildfake_labels = wildfake_data["wildfake_labels"]
    print(f"   ✅ WildFake: {len(wildfake_paths)} images")
    
    # ===== MODEL PATHS (from Person B) =====
    MODEL_PATHS = {
        "plain": "models/plain_model.pt",        # Person B provides this
        "augmented": "models/augmented_model.pt" # Person B provides this
    }
    
    # ===== MODEL LOADING FUNCTION (from Person B) =====
    def load_model(checkpoint_path):
        """
        Person B - FILL THIS IN with your model loading code.
        """
        # Example:
        # model = FakeDetector()
        # model.load_state_dict(torch.load(checkpoint_path))
        # return model
        pass
    
    # ===== RUN THE EVALUATION =====
    print("\n🧪 Running robustness evaluation...")
    
    results = run_robustness_table(
        model_paths=MODEL_PATHS,
        val_paths=val_paths,
        val_labels=val_labels,
        wildfake_paths=wildfake_paths,
        wildfake_labels=wildfake_labels,
        model_loader_fn=load_model
    )
    
    # ===== SAVE RESULTS =====
    with open("robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("✅ RESULTS SAVED TO: robustness_results.json")
    print("="*50)
    
    # ===== PRINT TABLE =====
    print("\n📊 ROBUSTNESS TABLE")
    print("=" * 70)
    print(f"{'Condition':<20} {'Plain Acc':<12} {'Plain AUC':<12} {'Aug Acc':<12} {'Aug AUC':<12}")
    print("-" * 70)
    
    conditions = list(results["plain"].keys())
    for condition in conditions:
        p_acc = results["plain"][condition]["accuracy"]
        p_auc = results["plain"][condition]["auc"]
        a_acc = results["augmented"][condition]["accuracy"]
        a_auc = results["augmented"][condition]["auc"]
        print(f"{condition:<20} {p_acc}%{'':<8} {p_auc:<12} {a_acc}%{'':<8} {a_auc:<12}")
    
    print("\n" + "="*70)
    print("✅ Evaluation complete! Send this table to Person D.")