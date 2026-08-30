# Robustness Evaluation Summary

Testing uses the **fixed-intensity conditions locked in Saturday night**
(see `test_conditions.py`) rather than every severity level — this keeps
Sunday's testing scope tight enough to actually finish in one day.

Locked conditions:
- JPEG quality = **30** (most degraded)
- Gaussian Blur σ = **2.0** (most degraded)
- Center Crop = **80%**
- Bonus (if time allows): stacked JPEG q30 + Blur σ2.0 ("social media simulation")

## Core Comparison: Plain vs. Augmented vs. Unseen-Generator

| Condition | Plain Model | Augmented Model | Notes |
|---|---|---|---|
| Clean | Acc: 87.29% / AUC: 0.9556 | Acc: 86.79% / AUC: 0.9658 | Baseline |
| JPEG q=30 | Acc: 85.99% / AUC: 0.9344 | Acc: 85.54% / AUC: 0.9525 | |
| Blur σ=2.0 | Acc: 89.61% / AUC: 0.9680 | Acc: 88.63% / AUC: 0.9694 | |
| Center Crop 80% | Acc: 84.68% / AUC: 0.9441 | Acc: 84.27% / AUC: 0.9486 | |
| Unseen Generator (WildFake) | ⏳ pending | ⏳ pending | Waiting on DDIM images |

## Plain-English Interpretation

<!-- D fills this in as C's numbers land (Sunday ~3:30pm onward, per
schedule). Keep it jargon-free — this is what a judge skimming quickly
will actually read closely. -->

**Does augmentation help?**
The picture here is nuanced rather than a clean win. Raw accuracy is
almost identical between the two models — in fact, the plain model is
marginally *higher* on every single condition (e.g. 89.61% vs 88.63% on
blur). However, AUC — which measures how well the model separates the
two classes regardless of the exact threshold — improved for the
augmented model across all four conditions (e.g. 0.9680 → 0.9694 on
blur, 0.9344 → 0.9525 on JPEG). This suggests training-time augmentation
made the model's underlying confidence scores more reliable and better
calibrated, even though it didn't necessarily produce more correct
hard predictions at the default threshold. In other words: augmentation
appears to improve the *quality* of the model's confidence, more than
its raw accuracy — the real test of practical robustness benefit will
be the unseen-generator (WildFake) result once that's in.

**Does it generalize to an unseen generator?**
<!-- e.g. "Performance on WildFake (a generator not seen in training)
dropped by X%, suggesting the model partly relies on generator-specific
artifacts rather than fully general signatures of AI generation." -->

**Biggest weak point:**
The plain model holds up well overall, staying between 85-90% accuracy
across every condition tested — no sharp cliffs like we initially
expected. Center Crop 80% caused the largest drop (84.68%), suggesting
the model relies somewhat on full-frame context rather than just local
detail. JPEG compression was close behind (85.99%). Interestingly, blur
actually performed *best* of all conditions (89.61%) — even slightly
above the clean baseline — which is counterintuitive enough that it's
worth mentioning directly rather than smoothing over. One possible
explanation: blur may remove high-frequency noise that occasionally
confuses the model, though this is worth flagging as an observation
rather than a confirmed explanation.

## Visualization
<!-- Optional: a bar chart comparing plain vs. augmented across the
conditions above tends to land better with judges than the raw table
alone — consider adding one before final submission. -->