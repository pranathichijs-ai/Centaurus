"""
from sklearn.metrics import roc_auc_score, accuracy_score
import torch

def evaluate(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            probs = torch.sigmoid(model(images)).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    return acc, auc
"""
