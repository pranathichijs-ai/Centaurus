"""
evaluate.py

Simple evaluate() function matching a standard PyTorch training loop
signature: takes a model and a val_loader (DataLoader), runs inference
over the whole validation set, and returns (accuracy, auc).

Uses the shared compute_metrics() from scoring.py under the hood, so
numbers stay consistent with everything else in the project (robustness
eval, error analysis, etc.) — just exposed as a simpler two-value return
for use directly inside a training loop.

Assumes:
  - val_loader yields (images, labels) batches, where images are already
    preprocessed tensors ready for the model, and labels are 0/1 tensors
    (0 = real, 1 = AI-generated).
  - model(images) returns raw logits or scores; adjust the
    `torch.sigmoid(...)` line below if your model already outputs
    probabilities, or uses a different output shape.

Requires: torch, scoring.py (for compute_metrics)
"""

import torch
from eval.scoring import compute_metrics


def evaluate(model, val_loader, device: str = None) -> tuple:
    """
    Runs the model over val_loader and returns (accuracy, auc).

    Args:
        model: a PyTorch model (already trained / in eval mode is set
               internally by this function)
        val_loader: PyTorch DataLoader yielding (images, labels) batches
        device: 'cuda' or 'cpu'; auto-detected if not passed

    Returns:
        (accuracy, auc) tuple of floats
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # If model outputs raw logits, convert to probabilities.
            # If model already outputs probabilities in [0, 1], skip this.
            scores = torch.sigmoid(outputs).squeeze(-1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_scores)

    accuracy = metrics["accuracy"]
    auc = metrics["auc"]

    return accuracy, auc


if __name__ == "__main__":
    # Smoke test with a fake model and fake DataLoader
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)

    # Fake model: single linear layer, outputs raw logits
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Fake dataset: 20 samples, 10 features each, binary labels
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,)).float()
    dataset = TensorDataset(X, y)
    val_loader = DataLoader(dataset, batch_size=4)

    model = FakeModel()
    acc, auc = evaluate(model, val_loader, device="cpu")
    print(f"Smoke test — accuracy: {acc:.3f}, auc: {auc:.3f}")
