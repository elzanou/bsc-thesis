# Pairwise — qwen2.5-omni-7b

| | |
|---|---|
| **Run ID** | `20260308_130908` |
| **Provider** | `qwen` |
| **Model** | `qwen2.5-omni-7b` |
| **Prompt hash** | `b53d4b6a` |


## Summary

Overall performance on the PAIRWISE task. Accuracy measures exact-match between predicted and ground truth answers. Precision, recall, and F1 are macro-averaged across all classes.

| Metric | Value |
|--------|------:|
| Total samples | 60 |
| Parse errors | 1 (1.7%) |
| Correct | 28 |
| **Accuracy** | **46.7%** |
| Precision (macro) | 0.463 |
| Recall (macro) | 0.471 |
| **F1 (macro)** | **0.443** |


## Per-Category Accuracy

Accuracy broken down by mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| harmony | 5 | 5 | 100.0% |
| tempo | 8 | 15 | 53.3% |
| articulation | 4 | 8 | 50.0% |
| technique | 4 | 8 | 50.0% |
| dynamics | 3 | 9 | 33.3% |
| pitch | 2 | 7 | 28.6% |
| rhythm_and_timing | 2 | 8 | 25.0% |


## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| B | 0.49 | 0.70 | 0.58 | 30 |
| A | 0.44 | 0.24 | 0.31 | 29 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| A | 16 | 27.1% | 29 | 49.2% |
| B | 43 | 72.9% | 30 | 50.8% |


## Confusion Matrices

### Overall

| Actual \ Predicted | A | B |
|---|---:|---:|
| **A** | 7 | 22 |
| **B** | 9 | 21 |

---

*Generated on 2026-03-08 20:59*
