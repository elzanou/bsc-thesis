# Evaluation Report — Pairwise

| | |
|---|---|
| **Run ID** | `20260308_153607` |
| **Provider** | `music_flamingo` |
| **Model** | `nvidia/music-flamingo-hf` |
| **Task** | `pairwise` |
| **Prompt hash** | `b53d4b6a` |


## Summary

| Metric | Value |
|--------|------:|
| Total samples | 60 |
| Parse errors | 0 (0.0%) |
| Correct | 33 |
| **Accuracy** | **55.0%** |
| Precision (macro) | 0.558 |
| Recall (macro) | 0.550 |
| **F1 (macro)** | **0.534** |


## Per-Category Analysis

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| pitch | 6 | 7 | 85.7% |
| harmony | 4 | 5 | 80.0% |
| articulation | 6 | 8 | 75.0% |
| technique | 4 | 8 | 50.0% |
| dynamics | 4 | 9 | 44.4% |
| tempo | 6 | 15 | 40.0% |
| rhythm_and_timing | 3 | 8 | 37.5% |


## Per-Class Precision / Recall / F1

| Category | Precision | Recall | F1 | Support |
|----------|----------:|-------:|---:|--------:|
| A | 0.54 | 0.73 | 0.62 | 30 |
| B | 0.58 | 0.37 | 0.45 | 30 |


## Prediction vs Ground Truth Distribution

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| A | 41 | 68.3% | 30 | 50.0% |
| B | 19 | 31.7% | 30 | 50.0% |


## Confusion Matrices

### Confusion Matrix

| Actual \ Predicted | A | B |
|---|---:|---:|
| **A** | 22 | 8 |
| **B** | 19 | 11 |

---

*Generated on 2026-03-08 17:41*
