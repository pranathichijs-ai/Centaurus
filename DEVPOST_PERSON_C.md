\# Augmentation \& Robustness Testing



\## Libraries Used

\- \*\*Albumentations (v1.3.1)\*\* - Primary library for image augmentation

\- \*\*PyTorch\*\* - For tensor operations and GPU acceleration

\- \*\*PIL (Pillow)\*\* - For image loading and preprocessing

\- \*\*NumPy\*\* - For array operations



\## Augmentation Pipeline Design



\### Training-Time Augmentation

We designed a stochastic augmentation pipeline to improve model robustness:



Pipeline Steps:

1\. Resize - All images resized to 224×224 (CLIP's expected input)

2\. Random Distortion - 80% chance of applying ONE of these:

&#x20;  - JPEG compression (quality 30-90)

&#x20;  - Gaussian blur (sigma 0.5-2.0)  

&#x20;  - Gaussian noise (variance 2.0-10.0)

&#x20;  - Color jitter (±20% brightness, contrast, saturation)

&#x20;  - Random crop (64-100% of image)

3\. Normalization - CLIP's exact values (mean: \[0.481, 0.458, 0.408], std: \[0.269, 0.261, 0.276])

4\. Convert to Tensor - PyTorch tensor format



Why This Design:

\- One distortion at a time - Real images typically have one type of degradation, not multiple

\- 80% probability - Allows some clean images to pass through for balanced learning

\- CLIP normalization - Matches the pretrained model's expectations



\### Test Conditions (Robustness Evaluation)

We evaluated both models (plain vs. augmented) under fixed conditions:



| Condition | Description |

|-----------|-------------|

| Clean | No distortion (baseline) |

| JPEG q30 | Heavy JPEG compression (quality=30) |

| Blur σ=2 | Heavy Gaussian blur |

| Crop 80% | Center crop to 80% of original |



\## Trade-offs and Key Insights



Accuracy vs. Robustness Trade-off:

\- The augmented model may sacrifice 1-2% accuracy on clean images

\- But gains 5-10% accuracy on compressed/blurred images

\- This trade-off is \*\*worth it\*\* for real-world applications where images often have compression artifacts



Why This Matters:

\- Social media platforms compress images heavily

\- User-uploaded photos vary in quality

\- Robustness is more valuable than perfect performance on clean data



\*\*Results Summary:\*\*

\[Fill in after Person D runs the table on Sunday]



\## What I Learned

Building this pipeline taught me:

1\. Augmentation is a powerful, low-cost way to improve model robustness

2\. The right augmentation strategy depends on your test conditions

3\. CLIP's specific normalization values are critical for performance



\## Future Improvements

\- Add more diverse augmentations (motion blur, lighting changes)

\- Experiment with different probabilities for each distortion

\- Test with more extreme compression levels

