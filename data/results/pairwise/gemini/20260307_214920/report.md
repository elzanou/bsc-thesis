# Pairwise — gemini-2.0-flash

| | |
|---|---|
| **Run ID** | `20260307_214920` |
| **Provider** | `gemini` |
| **Model** | `gemini-2.0-flash` |
| **Prompt hash** | `b53d4b6a` |


## Summary

Overall performance on the PAIRWISE task. Accuracy measures exact-match between predicted and ground truth answers. Precision, recall, and F1 are macro-averaged across all classes.

| Metric | Value |
|--------|------:|
| Total samples | 60 |
| Parse errors | 0 (0.0%) |
| Correct | 41 |
| **Accuracy** | **68.3%** |
| Precision (macro) | 0.684 |
| Recall (macro) | 0.683 |
| **F1 (macro)** | **0.683** |


## Per-Category Accuracy

Accuracy broken down by mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| articulation | 8 | 8 | 100.0% |
| tempo | 12 | 15 | 80.0% |
| dynamics | 7 | 9 | 77.8% |
| technique | 6 | 8 | 75.0% |
| pitch | 3 | 7 | 42.9% |
| harmony | 2 | 5 | 40.0% |
| rhythm_and_timing | 3 | 8 | 37.5% |


## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| A | 0.68 | 0.70 | 0.69 | 30 |
| B | 0.69 | 0.67 | 0.68 | 30 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| A | 31 | 51.7% | 30 | 50.0% |
| B | 29 | 48.3% | 30 | 50.0% |


## Confusion Matrices

### Overall

| Actual \ Predicted | A | B |
|---|---:|---:|
| **A** | 21 | 9 |
| **B** | 10 | 20 |

---

*Generated on 2026-03-08 20:59*
