import torch
import torch.nn as nn
!pip install open_clip_torch
import open_clip
from PIL import Image  # FIX 5: Image.open() is used in run_inference but was never imported

# Load CLIP
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

# Freeze CLIP
for param in clip_model.parameters():
    param.requires_grad = False

# Fakeimage Detector
class FakeDetector(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Linear(512, 1)
    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        return self.head(features).squeeze(-1)

# Function that takes in a folder containing images and outputs fake probability for each image
import json
import glob

def run_inference(image_dir, output_path, model):

    results = []

    for path in glob.glob(f"{image_dir}/*"):

        img = Image.open(path).convert("RGB")

        transformed = preprocess(img).unsqueeze(0).cuda()

        pred = torch.sigmoid(model(transformed)).item()

        results.append({
            "image_path": path,
            "pred": pred})

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

model = FakeDetector(clip_model).cuda()

!git clone https://github.com/pranathichijs-ai/Centaurus.git
!ls

!find /content/Centaurus -name "dataset.py"

from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

print(train_paths[:5])

import os; print(os.getcwd())

!unzip /content/archive.zip

with open('/content/Centaurus/aigc-detection3/train_splits.json') as f:
    data = json.load(f)
    train_paths = data["train_paths"]
    train_labels = data["train_labels"]

# fix Windows-style backslashes so paths work on Colab's Linux filesystem
train_paths = [p.replace("\\", "/") for p in train_paths]

train_dataset = ImageDataset(train_paths, train_labels, transform=preprocess)

import zipfile

with zipfile.ZipFile('/content/archive.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/data/cifake')

model.eval()  # tell the model we're testing, not training
run_inference('/content/data/cifake/train/REAL', '/content/test_output.json', model)

import os; print(os.path.exists('/content/test_output.json'))

import json; print(json.load(open('/content/test_output.json'))[:3]) #see the first few predictions. If you see real numbers between 0 and 1 next to file paths, it worked correctly.
