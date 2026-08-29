import glob
from sklearn.model_selection import train_test_split

real_paths = glob.glob("data/cifake/train/REAL/*.jpg")
fake_paths = glob.glob("data/cifake/train/FAKE/*.jpg")

all_paths = real_paths + fake_paths
all_labels = [0]*len(real_paths) + [1]*len(fake_paths)

print(f"Found {len(real_paths)} real images, {len(fake_paths)} fake images")

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.15, random_state=42, stratify=all_labels
)

print(f"Train: {len(train_paths)} images, Val: {len(val_paths)} images")
