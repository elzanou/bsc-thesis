# Evaluation Report — Pairwise

| | |
|---|---|
| **Run ID** | `20260308_165745` |
| **Provider** | `qwen` |
| **Model** | `qwen2.5-omni-7b` |
| **Task** | `pairwise` |
| **Prompt hash** | `b53d4b6a` |


## Summary

| Metric | Value |
|--------|------:|
| Total samples | 60 |
| Parse errors | 1 (1.7%) |
| Correct | 34 |
| **Accuracy** | **56.7%** |
| Precision (macro) | 0.591 |
| Recall (macro) | 0.579 |
| **F1 (macro)** | **0.564** |


## Per-Category Analysis

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| articulation | 6 | 8 | 75.0% |
| pitch | 5 | 7 | 71.4% |
| tempo | 9 | 15 | 60.0% |
| dynamics | 5 | 9 | 55.6% |
| technique | 4 | 8 | 50.0% |
| harmony | 2 | 5 | 40.0% |
| rhythm_and_timing | 3 | 8 | 37.5% |


## Per-Class Precision / Recall / F1

| Category | Precision | Recall | F1 | Support |
|----------|----------:|-------:|---:|--------:|
| B | 0.55 | 0.76 | 0.64 | 29 |
| A | 0.63 | 0.40 | 0.49 | 30 |


## Prediction vs Ground Truth Distribution

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| A | 19 | 32.2% | 30 | 50.8% |
| B | 40 | 67.8% | 29 | 49.2% |


## Confusion Matrices

### Confusion Matrix

| Actual \ Predicted | A | B |
|---|---:|---:|
| **A** | 12 | 18 |
| **B** | 7 | 22 |

---

*Generated on 2026-03-08 17:14*
