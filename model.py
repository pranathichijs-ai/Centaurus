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

# Function that takes in a folder containing images and output fake probability for each image 
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
        
#Dataloader that feed images in small batches to the model
from torch.utils.data import DataLoader

train_dataset = ImageDataset(train_paths, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#Run through FakeDetector
for images, labels in train_loader: 
  print(model(images.cuda())); 
  break

#Optimizer
optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-4) #we're only training the head
loss_fn = nn.BCEWithLogitsLoss() #tells PyTorch that we're doing binary classification. Real(0) vs Fake(1)?
for epoch in range(5): #go through the entire training dataset 5 times
    for images, labels in train_loader:#inside each epoch, we take 1 batch of 32 images
        images, labels = images.cuda(), labels.float().cuda() #move them onto GPU
        optimizer.zero_grad() #clears prev batch's calculations
        preds = model(images) #model makes predictions
        loss = loss_fn(preds, labels) #ask how wrong were those predictions
        loss.backward() #figures out how the head's parameters contributed to the error
        optimizer.step() #make changes to the head's parameters to improve future predictions
    print(f"Epoch {epoch}, loss: {loss.item()}") #the expected output is that as the epoch increases, the loss must decrease


