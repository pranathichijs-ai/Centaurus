# Error Analysis Note

## Overall Performance

The model achieves 90.1% accuracy and an AUC of 0.968 on the validation
set — AUC close to 1.0 means the model reliably ranks fake images as
"more fake" than real ones across almost any decision threshold, not
just at the specific cutoff used for accuracy. These numbers reflect
strong performance on data similar to what the model was trained on.

## False Positives (real images flagged as AI-generated)

The clearest examples all had prediction scores between 0.99 and
0.998 — extremely close to 1.0. The model wasn't uncertain on these; it
was **confidently wrong**. This is a more concerning failure mode than
an unsure guess, since it suggests these specific real images share
visual traits the model has learned to strongly associate with "fake"
(e.g. unusual lighting, compression artifacts, or a texture pattern
overrepresented in the fake training data).

## False Negatives (AI-generated images flagged as real)

The clearest examples had scores between 0.019 and 0.031 — again far
from the 0.5 decision boundary, meaning these were also confidently
wrong, just in the opposite direction. The model was sure these fakes
were real.

*(Note: the examples above are the model's most extreme errors, not a
random sample — they illustrate the model's worst mistakes, not its
typical behavior.)*

## Transform-Type Error Analysis

| Condition | Plain Errors | Plain Error Rate | Aug Errors | Aug Error Rate | Improvement |
|---|---|---|---|---|---|
| Clean | 1,912 | 12.75% | 1,422 | 9.48% | -3.27% |
| JPEG q=30 | 2,047 | 13.65% | 1,703 | 11.35% | -2.30% |
| Blur σ=2.0 | 1,547 | 10.31% | 1,361 | 9.07% | -1.24% |
| Center Crop 80% | 2,228 | 14.85% | 1,647 | 10.98% | -3.87% |
| WildFake | 15,645 | 49.38% | 15,023 | 47.42% | -1.96% |

**Key patterns:**
- Augmentation reduces the error rate on every single condition — no exceptions
- Center Crop 80% shows the biggest improvement from augmentation (-3.87% error rate) and is also the hardest condition for the plain model (14.85% error rate)
- JPEG q=30 is the hardest condition for the augmented model specifically (11.35% error rate)
- WildFake accounts for roughly half of all errors across the entire evaluation — by far the dominant source of mistakes, dwarfing every transformation condition combined

## WildFake (Unseen Generator) — Confidence Analysis

| Model | Accuracy | Confidently Wrong | Uncertain |
|---|---|---|---|
| Plain | 50.62% | 8,463 (54.1%) | 2,169 (13.9%) |
| Augmented | 52.58% | 8,596 (57.2%) | 1,997 (13.3%) |

This is the most important — and most concerning — pattern in the
whole analysis: **the models aren't just failing on WildFake, they're
confidently failing.** Over half of all WildFake errors (54-57%) carry
high-confidence scores, meaning the model isn't hedging or expressing
uncertainty when it encounters an unseen generator — it's making
confident, wrong calls. Only ~14% of errors show genuine uncertainty
(scores near 0.5).

**A closer look at the direction of these errors (Plain model, full
WildFake set — 6,000 real CelebA-HQ + 6,000 DDIM fake):**

| Model | Accuracy | AUC | Misclassified | False Positives (Real→Fake) | False Negatives (Fake→Real) |
|---|---|---|---|---|---|
| Plain | 47.12% | 0.4753 | 3,173/6,000 | 2,184 | 989 |
| Augmented | 47.97% | 0.4907 | 3,122/6,000 | 2,281 | 841 |

*(Note: these figures come from a separate WildFake-only evaluation
pass and differ slightly from the headline 50.62%/52.58% reported in
the main robustness table, which was computed on the full combined
WildFake test set used for the core comparison. Both point to the same
conclusion: performance is close to random chance.)*

**False positives dominate on WildFake** — the models consistently
mislabel real CelebA-HQ faces as fake, more often than they mislabel
actual DDIM-generated fakes as real. Example misclassifications (Plain
model), all confidently wrong (confidence 0.86–0.97 that a real photo
is fake):

- `img015707.jpg` → True: REAL, Predicted: FAKE, Confidence: 0.954
- `img004528.jpg` → True: REAL, Predicted: FAKE, Confidence: 0.969
- `img005530.jpg` → True: REAL, Predicted: FAKE, Confidence: 0.862

## Known Failure Modes / Trade-offs

**Robustness vs. generalization are separate problems.** Every
transformation condition (JPEG, blur, crop) stays in the 85-91%
accuracy range with augmentation improving each one. WildFake alone
drops to near-coin-flip territory (47-53%) for both models. Augmenting
against known transformations did not meaningfully help the model
generalize to an unseen generator — these are two different kinds of
robustness, and solving one does not solve the other.

**The model is overconfident specifically where it's least reliable.**
On in-distribution data (clean, transformed), high-confidence
predictions are usually correct. On WildFake, high confidence is
frequently *wrong* — the model has no internal signal that it's
operating outside its training distribution. This is a meaningful
practical risk for real-world deployment: a system that "knows what it
doesn't know" is far more useful than one that's equally (over)confident
everywhere.

**False positives on WildFake specifically target real human faces**
(CelebA-HQ) — the model appears to have learned patterns from its
training generators that don't transfer to recognizing *real* images
outside its training distribution either, not just fake ones. This
suggests the domain gap runs in both directions, not just "the model
doesn't recognize new fake styles."

## What We'd Improve With More Time

- Train on a broader and more diverse set of generators, rather than
  relying on augmentation-only robustness — augmentation improved
  robustness to transformations but didn't meaningfully close the
  generalization gap to an unseen generator (DDIM)
- Investigate confidence calibration specifically for out-of-distribution
  inputs — e.g. temperature scaling or an explicit "unfamiliar input"
  signal, so the model expresses appropriate uncertainty on unfamiliar
  generators rather than confidently guessing wrong
- Examine why false positives specifically cluster on real human faces
  in WildFake — this may point to the training data's real-image
  distribution being narrower than assumed (e.g. lacking sufficient
  diversity in lighting, ethnicity, or image style compared to
  CelebA-HQ)