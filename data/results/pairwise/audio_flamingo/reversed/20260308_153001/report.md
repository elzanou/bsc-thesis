# Evaluation Report — Pairwise

| | |
|---|---|
| **Run ID** | `20260308_153001` |
| **Provider** | `audio_flamingo` |
| **Model** | `nvidia/audio-flamingo-3-hf` |
| **Task** | `pairwise` |
| **Prompt hash** | `b53d4b6a` |


## Summary

| Metric | Value |
|--------|------:|
| Total samples | 60 |
| Parse errors | 4 (6.7%) |
| Correct | 32 |
| **Accuracy** | **53.3%** |
| Precision (macro) | 0.570 |
| Recall (macro) | 0.570 |
| **F1 (macro)** | **0.569** |


## Per-Category Analysis

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| pitch | 5 | 7 | 71.4% |
| tempo | 10 | 15 | 66.7% |
| articulation | 5 | 8 | 62.5% |
| rhythm_and_timing | 4 | 8 | 50.0% |
| dynamics | 4 | 9 | 44.4% |
| harmony | 2 | 5 | 40.0% |
| technique | 2 | 8 | 25.0% |


## Per-Class Precision / Recall / F1

| Category | Precision | Recall | F1 | Support |
|----------|----------:|-------:|---:|--------:|
| A | 0.58 | 0.62 | 0.60 | 29 |
| B | 0.56 | 0.52 | 0.54 | 27 |


## Prediction vs Ground Truth Distribution

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| A | 31 | 55.4% | 29 | 51.8% |
| B | 25 | 44.6% | 27 | 48.2% |


## Confusion Matrices

### Confusion Matrix

| Actual \ Predicted | A | B |
|---|---:|---:|
| **A** | 18 | 11 |
| **B** | 13 | 14 |

---

*Generated on 2026-03-08 17:41*
