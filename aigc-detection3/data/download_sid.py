from datasets import load_dataset
import os

os.makedirs("data/sid_set", exist_ok=True)

ds = load_dataset("saberzl/SID_Set", split="train", streaming=True)

count = 0
target = 3000
for example in ds:
    if count >= target:
        break
    img = example["image"]
    img = img.convert("RGB")
    label = example["label"]
    img.save(f"data/sid_set/{count}_label{label}.png")
    count += 1
    if count % 100 == 0:
        print(f"Downloaded {count} images...")

print("Done!")
