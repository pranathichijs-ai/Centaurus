import glob
import random
import json

random.seed(42)

# Gather all extracted fake images (DDIM) — adjust/add lines if Step 1 showed more folders
all_ddim_paths = glob.glob("data/wildfake/DDIM/DDIM/imgs_CC9K/*.png") + \
                  glob.glob("data/wildfake/DDIM/DDIM/imgs_bedroom/*.png")

# Gather all extracted real images (celebahq)
all_real_paths = glob.glob("data/wildfake/celebahq/celebahq/data1024x1024/*.jpg")

print(f"Total DDIM (fake) available: {len(all_ddim_paths)}")
print(f"Total celebahq (real) available: {len(all_real_paths)}")

# Balance to a manageable, matched subset size
subset_size = min(3000, len(all_real_paths), len(all_ddim_paths))

fake_wildfake_paths = random.sample(all_ddim_paths, subset_size)
real_wildfake_paths = random.sample(all_real_paths, subset_size)

wildfake_paths = real_wildfake_paths + fake_wildfake_paths
wildfake_labels = [0]*len(real_wildfake_paths) + [1]*len(fake_wildfake_paths)

print(f"Final wildfake_paths: {len(wildfake_paths)}")
print(f"Final wildfake_labels: {len(wildfake_labels)}")
wildfake_paths = [p.replace("\\", "/") for p in wildfake_paths]
# Save for handoff, same pattern as train_splits.json
with open("wildfake_splits.json", "w") as f:
    json.dump({"wildfake_paths": wildfake_paths, "wildfake_labels": wildfake_labels}, f)

print("Saved wildfake_splits.json")