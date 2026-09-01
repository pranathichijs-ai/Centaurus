# Robustness Evaluation Summary

Both `model_plain.pt` and `model_augmented.pt` were evaluated across four fixed conditions (Clean, JPEG_q30, Blur_sigma_2, Crop_80) plus the held-out WildFake unseen-generator test.

## Core Comparison: Plain vs. Augmented vs. Unseen-Generator

| Condition | Plain Model | Augmented Model |
|---|---|---|
| Clean | Acc: 87.25% / AUC: 0.9528 | Acc: 90.52% / AUC: 0.9691 |
| JPEG q=30 | Acc: 86.36% / AUC: 0.9388 | Acc: 88.65% / AUC: 0.9560 |
| Blur σ=2.0 | Acc: 89.69% / AUC: 0.9675 | Acc: 90.95% / AUC: 0.9689 |
| Center Crop 80% | Acc: 85.10% / AUC: 0.9442 | Acc: 88.93% / AUC: 0.9595 |
| Unseen Generator (WildFake) | Acc: 50.62% / AUC: 0.5091 | Acc: 52.58% / AUC: 0.5442 |

## Plain-English Interpretation

**Does augmentation help?**
Yes, clearly. Augmentation improves both accuracy and AUC across every single tested condition — largest gains on Crop (+3.83%) and Clean (+3.27%). This is a straightforward win, not a nuanced trade-off in raw accuracy.

**Does it generalize to an unseen generator?**
Only marginally. On WildFake (a generator and real-image source never seen during training), both models perform only slightly above random chance (50.62% plain, 52.58% augmented). Confidence-breakdown analysis shows both models are overconfident on these errors — the majority of wrong predictions are made with high confidence rather than genuine uncertainty, suggesting the models learned features specific to CIFAKE/SID_Set's particular data distribution rather than a fully generalizable notion of "real vs. fake."

**Biggest weak point:**
Generalization to unseen generators, not robustness to standard transformations. The model handles JPEG compression, blur, and cropping well — the real limitation is unseen-generator generalization, a known hard problem in AI-detection research broadly, not specific to this pipeline.

## Visualization
See `robustness_chart.png` (or the project README / Devpost submission) for a bar chart comparing plain vs. augmented model performance across all five conditions.

## Full Data
Complete per-image results: `robustness_results.json`, `wildfake_results.json` — see the Error Analysis Note (`eval/error_analysis_note.md`) for confidence-level breakdowns and misclassified examples.