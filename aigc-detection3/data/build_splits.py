import glob
from sklearn.model_selection import train_test_split

real_paths = glob.glob("data/cifake/train/REAL/*.jpg")
fake_paths = glob.glob("data/cifake/train/FAKE/*.jpg")

all_paths = real_paths + fake_paths
all_labels = [0]*len(real_paths) + [1]*len(fake_paths)
all_paths = [p.replace("\\", "/") for p in all_paths]
print(f"Found {len(real_paths)} real images, {len(fake_paths)} fake images")

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.15, random_state=42, stratify=all_labels
)

print(f"Train: {len(train_paths)} images, Val: {len(val_paths)} images")



sid_paths_raw = glob.glob("data/sid_set/*.png")
sid_labels_raw = [int(p.split("label")[1].split(".")[0]) for p in sid_paths_raw]

sid_paths = [p for p, l in zip(sid_paths_raw, sid_labels_raw) if l in (0, 1)]
sid_labels = [l for l in sid_labels_raw if l in (0, 1)]
sid_paths = [p.replace("\\", "/") for p in sid_paths]
print(f"Kept {len(sid_paths)} SID_Set images (real: {sid_labels.count(0)}, fake: {sid_labels.count(1)}), dropped {len(sid_paths_raw) - len(sid_paths)} tampered images")
