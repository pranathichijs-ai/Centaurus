import torch
import torch.nn as nn
import open_clip

#Load CLIP
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

#Freeze CLIP
for param in clip_model.parameters():
    param.requires_grad = False

#Fakeimage Detector
class FakeDetector(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Linear(512, 1)
    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        return self.head(features).squeeze(-1)
