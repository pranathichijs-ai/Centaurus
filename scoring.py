"""
scoring.py

Shared scoring function used across the project — by B (training/model
eval), C (robustness eval across transforms), and D (compiling the
results table / error analysis).

Takes predictions in the same format infer.py outputs (image_path, pred)
plus ground-truth labels, and returns standard classification metrics.
Designed to be called the same way regardless of which condition
(clean, JPEG q30, blur, unseen-generator, etc.) is being scored — so
everyone's numbers are computed consistently and are directly comparable
in the robustness table.

Requires: scikit-learn, numpy
"""

import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred_scores, threshold: float = 0.5) -> dict:
    """
    Compute standard classification metrics.

    Args:
        y_true: list/array of ground-truth labels (1 = AI-generated, 0 = real)
        y_pred_scores: list/array of predicted confidence scores in [0, 1]
                       (this is the "pred" field from infer.py's output)
        threshold: score >= threshold is classified as AI-generated (1)

    Returns:
        dict with accuracy, precision, recall, f1, auc, and confusion
        matrix components (tp, fp, tn, fn) — everything needed for both
        the robustness table and error analysis.
    """
    y_true = np.asarray(y_true)
    y_pred_scores = np.asarray(y_pred_scores)
    y_pred_labels = (y_pred_scores >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_labels),
        "precision": precision_score(y_true, y_pred_labels, zero_division=0),
        "recall": recall_score(y_true, y_pred_labels, zero_division=0),
        "f1": f1_score(y_true, y_pred_labels, zero_division=0),
    }

    # AUC requires both classes present in y_true
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_pred_scores)
    else:
        metrics["auc"] = None  # not computable with only one class

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels, labels=[0, 1]).ravel()
    metrics.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})

    return metrics


def score_from_files(predictions_json: str, labels_json: str, threshold: float = 0.5) -> dict:
    """
    Convenience wrapper: load predictions (infer.py output format) and
    ground-truth labels from disk, align them by image_path, and score.

    predictions_json format (list of dicts):
        [{"image_path": "...", "pred": 0.87}, ...]

    labels_json format (dict mapping image_path -> label):
        {"path/to/img1.jpg": 1, "path/to/img2.jpg": 0, ...}
        (1 = AI-generated, 0 = real)

    Returns the same dict as compute_metrics, plus a list of
    per-image records (for error analysis: which images were
    misclassified and how confidently).
    """
    with open(predictions_json, "r") as f:
        predictions = json.load(f)
    with open(labels_json, "r") as f:
        labels = json.load(f)

    y_true, y_pred_scores, records = [], [], []
    missing = []

    for entry in predictions:
        path = entry["image_path"]
        pred = entry["pred"]
        if path not in labels:
            missing.append(path)
            continue
        true_label = labels[path]
        y_true.append(true_label)
        y_pred_scores.append(pred)
        records.append({
            "image_path": path,
            "true_label": true_label,
            "pred_score": pred,
            "pred_label": int(pred >= threshold),
            "correct": int(pred >= threshold) == true_label,
        })

    if missing:
        print(f"Warning: {len(missing)} predictions had no matching label and were skipped.")

    metrics = compute_metrics(y_true, y_pred_scores, threshold=threshold)
    metrics["records"] = records
    metrics["num_scored"] = len(y_true)
    metrics["num_missing_labels"] = len(missing)

    return metrics


def get_misclassified(metrics: dict, kind: str = "false_positive", top_n: int = 10) -> list:
    """
    Pull out the most confidently-wrong examples for error analysis.

    kind: "false_positive" (real flagged as AI) or "false_negative"
          (AI flagged as real)
    top_n: how many examples to return, sorted by confidence (most
           confidently wrong first)
    """
    records = metrics.get("records", [])
    if kind == "false_positive":
        wrong = [r for r in records if r["true_label"] == 0 and r["pred_label"] == 1]
        wrong.sort(key=lambda r: r["pred_score"], reverse=True)
    elif kind == "false_negative":
        wrong = [r for r in records if r["true_label"] == 1 and r["pred_label"] == 0]
        wrong.sort(key=lambda r: r["pred_score"])
    else:
        raise ValueError("kind must be 'false_positive' or 'false_negative'")
    return wrong[:top_n]


def format_metrics_row(metrics: dict) -> str:
    """
    Format metrics as a single markdown table row: Acc / AUC — matches
    the format used in robustness_eval_template.md, so C can paste
    output straight into the table.
    """
    acc = f"{metrics['accuracy']:.3f}"
    auc = f"{metrics['auc']:.3f}" if metrics["auc"] is not None else "N/A"
    return f"Acc: {acc} / AUC: {auc}"


if __name__ == "__main__":
    # Smoke test with synthetic data
    np.random.seed(0)
    y_true = np.random.randint(0, 2, size=100)
    # simulate a decent-but-imperfect model
    y_pred_scores = np.clip(y_true + np.random.normal(0, 0.3, size=100), 0, 1)

    metrics = compute_metrics(y_true, y_pred_scores)
    print("Smoke test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nFormatted row:", format_metrics_row(metrics))