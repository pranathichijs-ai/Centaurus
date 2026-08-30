# error_analysis.py
# Dataset-source error analysis for Person A
# Fills in once B/C share model predictions

# Expected input format once B/C share results:
# predictions = list of dicts like:
# {"path": "data/sid_set/img001.jpg", "true_label": 0, "predicted_label": 1, "confidence": 0.82}

def categorize_by_source(image_path):
    """Identify which dataset an image path came from."""
    if "cifake" in image_path:
        return "CIFAKE"
    elif "sid_set" in image_path:
        return "SID_Set"
    elif "wildfake" in image_path:
        return "WildFake"
    return "Unknown"


def find_errors(predictions):
    """Split predictions into false positives and false negatives."""
    false_positives = [p for p in predictions if p["true_label"] == 0 and p["predicted_label"] == 1]
    false_negatives = [p for p in predictions if p["true_label"] == 1 and p["predicted_label"] == 0]
    return false_positives, false_negatives


def count_by_source(error_list):
    """Count how many errors come from each dataset source."""
    counts = {}
    for item in error_list:
        source = categorize_by_source(item["path"])
        counts[source] = counts.get(source, 0) + 1
    return counts


if __name__ == "__main__":
    # --- PLACEHOLDER: replace this with B/C's real predictions once available ---
    predictions = []

    if not predictions:
        print("No predictions loaded yet — waiting on B/C's model output.")
    else:
        fp, fn = find_errors(predictions)
        print(f"False positives: {len(fp)} — by source: {count_by_source(fp)}")
        print(f"False negatives: {len(fn)} — by source: {count_by_source(fn)}")