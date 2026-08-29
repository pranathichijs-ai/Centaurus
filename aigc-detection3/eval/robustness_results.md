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

The central table — shows the effect of training-time augmentation, plus
generalization to a generator never seen during training (WildFake).

| Condition | Plain Model (no aug.) | Augmented Model | Notes |
|---|---|---|---|
| Clean (no transform) | Acc: ___ / AUC: ___ | Acc: ___ / AUC: ___ | Baseline |
| JPEG q=30 | Acc: ___ / AUC: ___ | Acc: ___ / AUC: ___ | |
| Blur σ=2.0 | Acc: ___ / AUC: ___ | Acc: ___ / AUC: ___ | |
| Center Crop 80% | Acc: ___ / AUC: ___ | Acc: ___ / AUC: ___ | |
| Social Media Sim (bonus) | Acc: ___ / AUC: ___ | Acc: ___ / AUC: ___ | Only if time allows |
| Unseen Generator (WildFake) | Acc: ___ / AUC: ___ | Acc: ___ / AUC: ___ | Tests generalization, not just robustness |

*(Use `format_metrics_row()` from `scoring.py` to generate the "Acc: __ / AUC: __" strings directly from results — paste straight in.)*

## Plain-English Interpretation

<!-- D fills this in as C's numbers land (Sunday ~3:30pm onward, per
schedule). Keep it jargon-free — this is what a judge skimming quickly
will actually read closely. -->

**Does augmentation help?**
<!-- e.g. "The augmented model held onto X% more accuracy than the plain
model when images were blurred or compressed, showing training-time
augmentation meaningfully improves robustness." -->

**Does it generalize to an unseen generator?**
<!-- e.g. "Performance on WildFake (a generator not seen in training)
dropped by X%, suggesting the model partly relies on generator-specific
artifacts rather than fully general signatures of AI generation." -->

**Biggest weak point:**
<!-- Which single condition caused the largest accuracy drop, and why
might that be? -->

## Visualization
<!-- Optional: a bar chart comparing plain vs. augmented across the
conditions above tends to land better with judges than the raw table
alone — consider adding one before final submission. -->