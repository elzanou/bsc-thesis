# Pairwise — nvidia/audio-flamingo-3-hf

| | |
|---|---|
| **Run ID** | `20260307_154631` |
| **Provider** | `audio_flamingo` |
| **Model** | `nvidia/audio-flamingo-3-hf` |
| **Prompt hash** | `b53d4b6a` |


## Summary

Overall performance on the PAIRWISE task. Accuracy measures exact-match between predicted and ground truth answers. Precision, recall, and F1 are macro-averaged across all classes.

| Metric | Value |
|--------|------:|
| Total samples | 60 |
| Parse errors | 8 (13.3%) |
| Correct | 26 |
| **Accuracy** | **43.3%** |
| Precision (macro) | 0.487 |
| Recall (macro) | 0.488 |
| **F1 (macro)** | **0.481** |


## Per-Category Accuracy

Accuracy broken down by mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| technique | 5 | 8 | 62.5% |
| harmony | 3 | 5 | 60.0% |
| dynamics | 5 | 9 | 55.6% |
| tempo | 7 | 15 | 46.7% |
| articulation | 3 | 8 | 37.5% |
| pitch | 2 | 7 | 28.6% |
| rhythm_and_timing | 1 | 8 | 12.5% |


## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| A | 0.53 | 0.64 | 0.58 | 28 |
| B | 0.44 | 0.33 | 0.38 | 24 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| A | 34 | 65.4% | 28 | 53.8% |
| B | 18 | 34.6% | 24 | 46.2% |


## Confusion Matrices

### Overall

| Actual \ Predicted | A | B |
|---|---:|---:|
| **A** | 18 | 10 |
| **B** | 16 | 8 |

---

*Generated on 2026-03-08 20:59*
