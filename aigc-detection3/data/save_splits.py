import glob
import json
from sklearn.model_selection import train_test_split

real_paths = glob.glob("data/cifake/train/REAL/*.jpg")
fake_paths = glob.glob("data/cifake/train/FAKE/*.jpg")

all_paths = real_paths + fake_paths
all_labels = [0]*len(real_paths) + [1]*len(fake_paths)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.15, random_state=42, stratify=all_labels
)

with open("train_splits.json", "w") as f:
    json.dump({"train_paths": train_paths, "train_labels": train_labels}, f)

with open("val_splits.json", "w") as f:
    json.dump({"val_paths": val_paths, "val_labels": val_labels}, f)

print(f"Saved {len(train_paths)} train paths -> train_splits.json")
print(f"Saved {len(val_paths)} val paths -> val_splits.json")
