# eval/infer.py
import torch
import json
import glob
from PIL import Image

def run_inference(image_dir, output_path, model, preprocess):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    results = []
    for path in glob.glob(f"{image_dir}/*"):
        img = Image.open(path).convert("RGB")
        transformed = preprocess(img).unsqueeze(0).to(device)
        pred = torch.sigmoid(model(transformed)).item()
        results.append({"image_path": path, "pred": pred})

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)