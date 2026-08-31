# Person C - Augmentation & Robustness Testing

## Libraries Used
- **Albumentations (v1.3.1)** - Image augmentation
- **PyTorch** - Deep learning framework
- **PIL (Pillow)** - Image loading
- **NumPy** - Array operations
- **scikit-learn** - Metrics (accuracy, AUC)
- **open_clip_torch** - CLIP model

## Augmentation Pipeline

### Training Augmentation
- Resize to 224×224 (CLIP input size)
- One-of distortion (80% probability):
  - JPEG compression (quality 30-90)
  - Gaussian blur (sigma 0.5-2.0)
  - Gaussian noise (var 2.0-10.0)
  - Color jitter (±20%)
  - Random crop (64-100%)
- CLIP normalization (mean: [0.481, 0.458, 0.408], std: [0.269, 0.261, 0.276])

### Test Conditions
1. Clean (baseline)
2. JPEG q30 (heavy compression)
3. Blur σ2.0 (heavy blur)
4. Crop 80% (center crop)
5. WildFake (unseen generator)

## Results

| Condition | Plain Acc | Plain AUC | Aug Acc | Aug AUC |
|-----------|-----------|-----------|---------|---------|
| Clean | 87.25% | 0.9528 | 90.52% | 0.9691 |
| JPEG_q30 | 86.36% | 0.9388 | 88.65% | 0.9560 |
| Blur_sigma_2 | 89.69% | 0.9675 | 90.95% | 0.9689 |
| Crop_80 | 85.10% | 0.9442 | 88.93% | 0.9595 |
| WildFake | 50.62% | 0.5091 | 52.58% | 0.5442 |

## Key Findings
- **Augmentation improves ALL conditions**
- Best improvement: Crop (+3.83%), Clean (+3.27%)
- AUC consistently higher for augmented model
- WildFake remains challenging (near random chance)

## Trade-offs
- Augmented model sacrifices minimal clean accuracy for significant robustness gains
- Training with augmentation adds ~20-30% training time
- One distortion at a time (not stacked) - more realistic

## Conclusion
Augmentation is effective for improving model robustness, but unseen generators require further research.
