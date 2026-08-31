# Robust Detection of AI-Generated Images Under Real-World Transformations

## Project Overview
AI-generated images are increasingly used for misinformation, impersonation, and fraud, and simple compression, cropping, or reposting can defeat naive detectors. This project builds a binary real-vs-synthetic image classifier designed to stay accurate not just on clean images, but after common real-world transformations — the kind of degradation images actually undergo when shared online.

## Problem Addressed
Most AI-image detectors are evaluated only on clean, unmodified images, but images shared on social platforms are routinely compressed, resized, cropped, or blurred before anyone sees them. A detector that only works on pristine images provides a false sense of security. This project specifically targets robustness — testing and training the model against JPEG compression, blur, and cropping, and validating generalization on an entirely unseen generator (WildFake) that the model never saw during training.

## Approach
- **Model:** CLIP ViT-B/32 as a frozen backbone, with a small trainable linear classifier head (`nn.Linear(512, 1)`) on top. CLIP has already been trained on hundreds of millions of images and has learned general visual understanding — starting from CLIP means borrowing "eyes that already work" instead of teaching a model to see from scratch within a hackathon timeframe.
- **Why frozen:** `requires_grad = False` on all CLIP parameters preserves its existing knowledge; only the new classifier head is trained. This keeps training fast and avoids damaging CLIP's pretrained understanding.
- **How it works:** CLIP encodes each image into 512 numbers describing what it "noticed" (`clip_model.encode_image(images)`); the linear head takes those 512 numbers and outputs one number — the model's confidence that the image is fake.
- **Training:** two model variants were trained — a plain baseline (`model_plain.pt`) and an augmented version (`model_augmented.pt`, trained with the augmentation pipeline below applied), to compare robustness with and without training-time augmentation. Optimizer: Adam (`lr=1e-4`, applied only to the classifier head, not the frozen CLIP backbone). Loss: `BCEWithLogitsLoss`. 5 epochs, batch size 32.
- **Augmentation strategy (training):** for each training image, exactly one random distortion is applied 80% of the time (`A.OneOf(..., p=0.8)`), chosen from: JPEG compression (quality 30–90), Gaussian blur (σ 0.5–2.0), Gaussian noise, color jitter, or random-resized crop (scale 0.64–1.0). All images are resized to 224×224 and normalized using CLIP's exact mean/std before being passed to the model.
- **Robustness evaluation conditions (testing):** four fixed conditions are used to measure robustness after training — Clean (no distortion), JPEG at a fixed quality of 30, Gaussian blur at a fixed σ of 2.0, and a fixed 80%-scale center crop. Unlike training (which picks one random distortion), each test condition applies one specific, non-random setting so results are directly comparable across images.

## Datasets Used
| Dataset | Purpose | Notes |
|---|---|---|
| CIFAKE | Main real/fake training data | Full standard dataset (~120k images: 100k train + 20k test, balanced real/fake) |
| SID_Set | Merged with CIFAKE to expand combined training set | 3,000-image subset; "tampered" label excluded to keep task strictly binary; 1,989 usable after filtering (1,001 real, 988 fake) |
| WildFake | Held-out "unseen generator" robustness test set — never used in training | 3,000 real (celebahq) + 3,000 fake (DDIM diffusion subset), balanced, seed=42 |

Combined training set after merging CIFAKE + SID_Set: 86,690 train images / 15,299 validation images (85/15 stratified split).

## Development Tools
- GitHub — shared repo for version control and file handoffs
- VS Code — shared development environment across the team
- Google Drive — sharing large files (raw datasets, trained models) too large for git
- Google Colab — used by Person B to run/train the model (GPU access)

## Models / APIs / Libraries
- scikit-learn — `train_test_split` for stratified train/val splitting
- open_clip_torch — CLIP ViT-B/32 pretrained backbone (`open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')`)
- PyTorch — model definition and training loop
- albumentations — augmentation pipeline (training-time distortions and fixed-condition robustness testing)

## Setup & Installation
```bash
git clone https://github.com/pranathichijs-ai/Centaurus.git
cd aigc-detection3
pip install -r requirements.txt
```
> Note: raw dataset images are not included in the repo (too large for git). Download from [Drive link — CONFIRM final link] and extract into `data/` — see folder structure below.

Required folder structure after extraction:
```
data/
├── cifake/train/REAL/*.jpg
├── cifake/train/FAKE/*.jpg
├── sid_set/*.png
└── wildfake/
    ├── celebahq/celebahq/data1024x1024/*.jpg
    └── DDIM/DDIM/imgs_CC9K/*.png, imgs_bedroom/*.png
```

## Steps to Reproduce Results
1. Download and extract datasets from [Drive link] into `data/` (structure above).
2. From the project root, generate the train/val/test splits:
   ```bash
   python data/save_splits.py
   python data/build_wildfake_splits.py
   ```
   This produces `data/train_splits.json`, `data/val_splits.json`, `data/wildfake_splits.json`.
3. (Optional sanity check) Confirm the augmentation pipeline integrates correctly with the dataset loader on a small placeholder sample:
   ```bash
   python train/train_integration.py
   ```
4. Train the model — plug `get_train_augmentation()` from `augmentations.py` into Person A's `ImageDataset`, then run the full training loop over `train_splits.json`:
   <!-- CONFIRM WITH B — actual training script name/command; train_integration.py is only a small-scale integration smoke test, not the full training run -->
5. Evaluate — `evaluate(model, val_loader)` (in `eval/evaluate.py`) returns `(accuracy, auc)` using `eval/scoring.py`'s `compute_metrics()`; call this from inside the training/eval script with a real model and `val_loader` built from `val_splits.json`.
6. Robustness evaluation (clean vs. transformed conditions) — fill in model paths, then run:
   ```bash
   python run_robustness_table.py
   ```

## Running Inference
`eval/infer.py` is not yet implemented (currently empty) — this section will be completed once the script is written.
<!-- WAITING ON B/D — script is a placeholder file with no code yet -->

## Robustness Evaluation Summary
Both `model_plain.pt` and `model_augmented.pt` are evaluated across four fixed conditions: Clean, JPEG_q30, Blur_sigma_2, and Crop_80 (see Approach above for exact settings). Run via:
```bash
python run_robustness_table.py
```
<!-- OWNED BY C, cross-checked by D — full results table (accuracy/AUC per condition, per model variant) -->

## Error Analysis
<!-- IN PROGRESS (A) — waiting on B/C's misclassified prediction examples, re-run needed against corrected splits -->

## Limitations & Future Work
- WildFake evaluation only tests one unseen generator family (DDIM diffusion) — future work could test against a broader range of generators (GANs, other diffusion variants, commercial tools like Midjourney) to check how well robustness generalizes beyond what was tested here.
- SID_Set contributes a relatively small portion of the combined training set (1,989 of ~102,000 images) — a more balanced mix across datasets might reduce any bias toward CIFAKE's particular generation style.
- Robustness was tested against a fixed set of transform conditions (JPEG q30, blur σ2.0, crop 80%) — real-world reposting often applies multiple transformations in sequence (e.g. compress AND crop), which wasn't tested here.
- With more time, testing intermediate transform intensities (not just one fixed severity per condition) would show whether robustness degrades gradually or falls off a cliff at some threshold.

<!-- CONFIRM WITH C — the project brief calls for augmentation probability values and parameter ranges to genuinely match the competition's official transform table; this hasn't been explicitly cross-checked in this document -->

## Team Contributions
| Member | Contribution |
|---|---|
| A (Jeny) | Data pipeline: CIFAKE/SID_Set/WildFake download, preprocessing, train/val/test splitting, bug fixes (cross-platform paths, dataset merge) |
| B (darsini) | Model architecture (CLIP ViT-B/32 + linear probe), training |
| C (camelia) | Augmentation pipeline, robustness testing across transform conditions |
| D (pranathi) | Coordination, evaluation metrics, Devpost compilation |

## Demo Video
[Demo Video](https://youtu.be/dxHURhy99_g)

