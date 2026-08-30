# Pairwise — nvidia/music-flamingo-hf

| | |
|---|---|
| **Run ID** | `20260307_173259` |
| **Provider** | `music_flamingo` |
| **Model** | `nvidia/music-flamingo-hf` |
| **Prompt hash** | `b53d4b6a` |


## Summary

Overall performance on the PAIRWISE task. Accuracy measures exact-match between predicted and ground truth answers. Precision, recall, and F1 are macro-averaged across all classes.

| Metric | Value |
|--------|------:|
| Total samples | 60 |
| Parse errors | 0 (0.0%) |
| Correct | 22 |
| **Accuracy** | **36.7%** |
| Precision (macro) | 0.356 |
| Recall (macro) | 0.367 |
| **F1 (macro)** | **0.355** |


## Per-Category Accuracy

Accuracy broken down by mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| rhythm_and_timing | 4 | 8 | 50.0% |
| tempo | 7 | 15 | 46.7% |
| pitch | 3 | 7 | 42.9% |
| articulation | 3 | 8 | 37.5% |
| dynamics | 3 | 9 | 33.3% |
| harmony | 1 | 5 | 20.0% |
| technique | 1 | 8 | 12.5% |


## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| A | 0.39 | 0.50 | 0.44 | 30 |
| B | 0.32 | 0.23 | 0.27 | 30 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| A | 38 | 63.3% | 30 | 50.0% |
| B | 22 | 36.7% | 30 | 50.0% |


## Confusion Matrices

### Overall

| Actual \ Predicted | A | B |
|---|---:|---:|
| **A** | 15 | 15 |
| **B** | 23 | 7 |

---

*Generated on 2026-03-08 20:59*
