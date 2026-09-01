# Robust Detection of AI-Generated Images Under Real-World Transformations

## Project Overview
AI-generated images are increasingly used for misinformation, impersonation, and fraud, and simple compression, cropping, or reposting can defeat naive detectors. This project builds a binary real-vs-synthetic image classifier designed to stay accurate not just on clean images, but after common real-world transformations — the kind of degradation images actually undergo when shared online.

## Problem Addressed
Most AI-image detectors are evaluated only on clean, unmodified images, but images shared on social platforms are routinely compressed, resized, cropped, or blurred before anyone sees them. A detector that only works on pristine images provides a false sense of security. This project specifically targets robustness — testing and training the model against JPEG compression, blur, and cropping, and validating generalization on an entirely unseen generator (WildFake) that the model never saw during training.

## Approach
- **Model:** CLIP ViT-B/32 as a frozen backbone, with a small trainable linear classifier head (`nn.Linear(512, 1)`) on top. CLIP has already been trained on hundreds of millions of images and has learned general visual understanding — starting from CLIP means borrowing "eyes that already work" instead of teaching a model to see from scratch within a hackathon timeframe. CLIP ViT-B/32 has ~150M parameters, well within the competition's <2B parameter constraint.
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
- Albumentations 1.3.1 — image augmentation
- PyTorch 2.11.0 — tensor operations, model definition and training loop
- Pillow (PIL) 11.3.0 — image loading
- NumPy 2.1.3 — array operations
- Matplotlib 3.10.0 — visualization
- scikit-learn 1.6.1 — metrics (accuracy, AUC) and `train_test_split` for stratified train/val splitting
- open_clip_torch 3.3.0 — CLIP ViT-B/32 pretrained backbone (`open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')`)

## Setup & Installation
```bash
git clone https://github.com/pranathichijs-ai/Centaurus.git
cd aigc-detection3
pip install -r requirements.txt
```
Datasets (CIFAKE, SID_Set, WildFake) are too large for git and are not included in this repo. Download from the original public sources:
- CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- SID_Set: https://huggingface.co/datasets/saberzl/SID_Set
- WildFake: https://modelscope.cn/datasets/hy2628982280/WildFake/summary (use the page's translation button)

> Team members: a pre-packaged zip of the exact subset used is also available via the shared Drive folder — ask in the team channel for the link, faster than downloading and resampling from scratch.

Place downloaded data under `data/` matching the folder structure below.

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
1. Download and extract datasets into `data/` (see Setup & Installation above for sources and folder structure).
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
4. Train the model — run in Google Colab (no standalone local training script; this is the exact sequence used): load and freeze CLIP → define the `FakeDetector` classifier head → build `train_loader` from `data/train_splits.json` (CIFAKE+SID_Set combined) → train 5 epochs → save `model_plain.pt` → rebuild the loader using C's `get_train_augmentation()` as the transform → train 5 more epochs → save `model_augmented.pt`. Full notebook: https://colab.research.google.com/drive/1dGEZukDrf7H1MLLpU34JwPIOWLmeDAGy?usp=sharing
5. Evaluate — `evaluate(model, val_loader)` (in `eval/evaluate.py`) returns `(accuracy, auc)` using `eval/scoring.py`'s `compute_metrics()`; call this from inside the training/eval script with a real model and `val_loader` built from `val_splits.json`. To test `evaluate.py` standalone (e.g. its built-in smoke test), run it as a module, **not** as a direct file path:
   ```bash
   python -m eval.evaluate
   ```
   > Running `python eval/evaluate.py` directly will fail with `ModuleNotFoundError: No module named 'eval'` — this is because `evaluate.py` uses a package-style import (`from eval.scoring import compute_metrics`), which only resolves correctly when Python treats `eval` as a package relative to the project root. The `-m` flag does this correctly; running the file path directly does not.
6. Robustness evaluation (clean vs. transformed conditions) — fill in model paths, then run:
   ```bash
   python run_robustness_table.py
   ```
   > Note: the `__main__` block's `load_model()` function is currently a placeholder stub (`pass`). Before running standalone, replace it with real model-loading code, e.g.:
   > ```python
   > def load_model(checkpoint_path):
   >     model = FakeDetector(clip_model)
   >     model.load_state_dict(torch.load(checkpoint_path))
   >     return model
   > ```
   > Also fill in `VAL_PATHS`/`VAL_LABELS` (from `data/val_splits.json`) and `MODEL_PATHS`. The actual robustness results in this README were generated by calling `run_robustness_table()` directly (with real model loading) from a Colab notebook, not by running this file as-is.
7. Generate the robustness comparison chart:
```bash
   python generate_robustness_chart.py
```
   This produces `eval/robustness_chart.png`, a bar chart comparing plain vs. augmented model accuracy across all 5 conditions (used in the Robustness Evaluation Summary below).
   > Note: `generate_robustness_chart.py` currently uses the final confirmed
   > results as hardcoded values, rather than reading them from
   > `robustness_results.json` directly.

## Running Inference
`eval/infer.py` provides `run_inference(image_dir, output_path, model, preprocess)`, which loops over every image in `image_dir`, runs it through the model, and writes a JSON file of `{"image_path": ..., "pred": ...}` records — matching the required deliverable format exactly.

Example usage:
```python
from eval.infer import run_inference
run_inference("path/to/images", "results.json", model, preprocess)
```
Output format:
```json
[
  {"image_path": "path/to/img1.jpg", "pred": 0.87},
  {"image_path": "path/to/img2.jpg", "pred": 0.12}
]
```
> The script auto-detects GPU availability and falls back to CPU if none is present, skips non-image files, and continues past any unreadable images rather than crashing — so it runs on any machine, not just the GPU environment used for development.

## Robustness Evaluation Summary
Both `model_plain.pt` and `model_augmented.pt` were evaluated across four fixed conditions (Clean, JPEG_q30, Blur_sigma_2, Crop_80) plus the held-out WildFake unseen-generator test:

| Condition | Plain Acc | Plain AUC | Augmented Acc | Augmented AUC |
|---|---|---|---|---|
| Clean | 87.25% | 0.9528 | 90.52% | 0.9691 |
| JPEG_q30 | 86.36% | 0.9388 | 88.65% | 0.9560 |
| Blur_sigma_2 | 89.69% | 0.9675 | 90.95% | 0.9689 |
| Crop_80 | 85.10% | 0.9442 | 88.93% | 0.9595 |
| WildFake (unseen generator) | 50.62% | 0.5091 | 52.58% | 0.5442 |

**Key findings:**
- Augmentation improves accuracy and AUC across every standard transform condition — largest gains on Crop (+3.83%) and Clean (+3.27%).
- Blur is the easiest condition for both models; JPEG compression and cropping are harder.
- **WildFake results are only slightly above random chance (~50-53%)** — both models struggle to generalize to an unseen generator (DDIM) and unseen real faces (CelebA-HQ), even after training on the combined CIFAKE+SID_Set dataset. Augmentation provides a modest improvement (+1.96%), not enough to close the domain gap. This is a genuine limitation, not a bug — see Limitations & Future Work.

Full results: `robustness_results.json`, `wildfake_results.json`.

## Error Analysis

### CIFAKE/SID_Set — dataset-source clustering
On the standard validation set, sample misclassified examples from the **augmented model** show errors are not isolated to one dataset:

**False positives (real → fake), all with very high confidence:**
```
data/cifake/train/REAL/2561.jpg          → conf: 0.998
data/cifake/train/REAL/2426 (3).jpg      → conf: 0.994
data/sid_set/1496_label0.png             → conf: 0.993
```

**False negatives (fake → real), all with very high confidence:**
```
data/cifake/train/FAKE/1696 (10).jpg     → conf: 0.019 (i.e. 98% confident it's real)
data/cifake/train/FAKE/4221 (2).jpg      → conf: 0.024
data/cifake/train/FAKE/1735 (10).jpg     → conf: 0.031
```

**Pattern:** in this sample, SID_Set appears among false positives (1 of 3) but not false negatives; CIFAKE appears in both categories. Sample size is small (6 examples) so this isn't a definitive dataset-source trend — but notably, like the WildFake errors below, the model is consistently *confidently* wrong rather than uncertain, suggesting overconfidence is a pattern across the whole system, not specific to any one dataset.

### WildFake — unseen generator
On the WildFake unseen-generator test set (31,682 images, retrained CIFAKE+SID_Set models), both models perform only slightly above random chance:

| Model | Accuracy | Avg Confidence (Correct) | Avg Confidence (Wrong) |
|---|---|---|---|
| Plain | 50.62% | 0.4849 | 0.4638 |
| Augmented | 52.58% | 0.5446 | 0.5282 |

**Confidence breakdown of wrong predictions:**
| Model | Very Confidently Wrong | Moderately Confident Wrong | Uncertain |
|---|---|---|---|
| Plain | 8,463 (54.1%) | 5,013 | 2,169 (13.9%) |
| Augmented | 8,596 (57.2%) | 4,430 | 1,997 (13.3%) |

**Pattern:** both models are **overconfident** on WildFake — the majority of wrong predictions (54-57%) are made with high confidence rather than near the 0.5 decision boundary, meaning the models "think they know" but are frequently wrong. This suggests the models learned features specific to CIFAKE/SID_Set's particular data distribution rather than a generalizable notion of "real vs. fake," even after including SID_Set in training.

Full results: `wildfake_error_analysis.json`, `complete_error_analysis.json`. CIFAKE/SID_Set examples above from `misclassified_examples.json`.

## Limitations & Future Work
- **The model does not generalize well to unseen generators or unseen real-image distributions, even after retraining on the combined CIFAKE+SID_Set dataset.** WildFake evaluation (DDIM-generated fakes + CelebA-HQ real faces, neither seen during training) shows only ~50-53% accuracy for both the plain and augmented models — barely above chance. This is a genuine limitation, not a bug.
- **Both models are overconfident on WildFake, not just wrong.** 54-57% of wrong predictions are made with high confidence (outside the 0.2-0.8 range) rather than near the uncertain 0.5 boundary — the models "think they know" but are frequently mistaken, suggesting they've learned features specific to CIFAKE/SID_Set's particular data distribution rather than a truly generalizable notion of "real vs. fake."
- **Deployment implication:** because the model is confidently wrong rather than uncertain on unfamiliar generators, a naive deployment — auto-flagging anything above a 0.5 threshold — would silently mislabel real content with high apparent certainty. A safer deployment would treat confidence as unreliable outside the training distribution: route borderline or unfamiliar-looking cases to human review rather than auto-flagging or auto-removing, especially content resembling generators or subjects not represented in training.
- Future work: since more training data (SID_Set) alone did not meaningfully close this gap, future work should explore training on a wider range of generator types directly (not just adding more real-image diversity), and/or techniques specifically aimed at reducing overconfidence on out-of-distribution inputs (e.g. calibration methods).
- Robustness was tested against a fixed set of transform conditions (JPEG q30, blur σ2.0, crop 80%) at a single severity level each — real-world images often undergo multiple stacked transformations at once (e.g. a screenshot that's also been recompressed), which was only explored, if at all, via an optional combined "social media simulation" condition, not systematically.
- With more time, testing intermediate transform intensities (not just one fixed severity per condition) would show whether robustness degrades gradually or falls off a cliff at some threshold.
- With more time, frequency-domain features alongside CLIP's visual features could help — GAN/diffusion artifacts often show up more clearly in the frequency spectrum than in raw pixels, and might improve generalization to unseen generators.
- **Trade-off (confirmed by C):** the augmented model sacrifices ~1-2% clean accuracy compared to the plain baseline, but gains 3-5% accuracy on distorted images — a worthwhile trade for real-world robustness. Training time increases ~20-30% with augmentation enabled.

## Team Contributions
| Member | Contribution |
|---|---|
| A (Jeny) | Dataset acquisition and preprocessing (CIFAKE, SID_Set, WildFake); data-loading pipeline; train/val/test split generation; bug fixes (cross-platform paths, dataset merge); dataset-source error analysis; README |
| B (darsini) | Model architecture (CLIP ViT-B/32 + linear classifier head); training both plain and augmented models; inference script; confidence-based error analysis |
| C (camelia) | Augmentation pipeline (Albumentations); robustness evaluation across fixed-intensity test conditions; transform-type error analysis |
| D (pranathi) | Shared scoring infrastructure; locked test-condition parameters; robustness results compilation; combined error analysis; Devpost writeup coordination |

## Demo Video
https://youtu.be/LrPmf5uXT50