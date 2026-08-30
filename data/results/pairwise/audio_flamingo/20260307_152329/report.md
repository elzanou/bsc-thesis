# Pairwise — nvidia/audio-flamingo-3-hf

| | |
|---|---|
| **Run ID** | `20260307_152329` |
| **Provider** | `audio_flamingo` |
| **Model** | `nvidia/audio-flamingo-3-hf` |
| **Prompt hash** | `b53d4b6a` |


## Summary

Overall performance on the PAIRWISE task. Accuracy measures exact-match between predicted and ground truth answers. Precision, recall, and F1 are macro-averaged across all classes.

| Metric | Value |
|--------|------:|
| Total samples | 60 |
| Parse errors | 7 (11.7%) |
| Correct | 19 |
| **Accuracy** | **31.7%** |
| Precision (macro) | 0.349 |
| Recall (macro) | 0.361 |
| **F1 (macro)** | **0.347** |


## Per-Category Accuracy

Accuracy broken down by mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| technique | 4 | 8 | 50.0% |
| tempo | 7 | 15 | 46.7% |
| dynamics | 3 | 9 | 33.3% |
| articulation | 2 | 8 | 25.0% |
| harmony | 1 | 5 | 20.0% |
| pitch | 1 | 7 | 14.3% |
| rhythm_and_timing | 1 | 8 | 12.5% |


## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| A | 0.38 | 0.50 | 0.43 | 26 |
| B | 0.32 | 0.22 | 0.26 | 27 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| A | 34 | 64.2% | 26 | 49.1% |
| B | 19 | 35.8% | 27 | 50.9% |


## Confusion Matrices

### Overall

| Actual \ Predicted | A | B |
|---|---:|---:|
| **A** | 13 | 13 |
| **B** | 21 | 6 |

---

*Generated on 2026-03-08 20:59*
