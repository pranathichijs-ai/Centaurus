import torch
import torch.nn as nn
import open_clip

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

# Function that takes in a folder of images and outputs fake probability for each
import json
import glob
from PIL import Image

def run_inference(image_dir, output_path, model):
    results = []
    for path in glob.glob(f"{image_dir}/*"):
        img = Image.open(path).convert("RGB")
        transformed = preprocess(img).unsqueeze(0).cuda()
        pred = torch.sigmoid(model(transformed)).item()
        results.append({"image_path": path, "pred": pred})

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
