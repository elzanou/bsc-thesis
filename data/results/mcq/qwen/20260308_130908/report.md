# Mcq — qwen2.5-omni-7b

| | |
|---|---|
| **Run ID** | `20260308_130908` |
| **Provider** | `qwen` |
| **Model** | `qwen2.5-omni-7b` |
| **Prompt hash** | `ac297321` |


## Summary

Overall performance on the MCQ task. Accuracy measures exact-match between predicted and ground truth answers. Precision, recall, and F1 are macro-averaged across all classes.

| Metric | Value |
|--------|------:|
| Total samples | 120 |
| Parse errors | 1 (0.8%) |
| Correct | 36 |
| **Accuracy** | **30.0%** |
| Precision (macro) | 0.224 |
| Recall (macro) | 0.173 |
| **F1 (macro)** | **0.129** |


## Results by Audio Setting

Single-audio samples provide only the student recording. Double-audio samples include both a reference performance and the student recording, separated by a beep.

| Setting | Samples | Accuracy |
|---------|--------:|---------:|
| Single audio | 48 | 33.3% |
| Double audio | 71 | 28.2% |


## Per-Category Accuracy

Accuracy broken down by mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| no_mistake | 32 | 32 | 100.0% |
| articulation | 2 | 9 | 22.2% |
| dynamics | 1 | 9 | 11.1% |
| tempo | 1 | 20 | 5.0% |
| pitch | 0 | 13 | 0.0% |
| rhythm_and_timing | 0 | 14 | 0.0% |
| technique | 0 | 13 | 0.0% |
| harmony | 0 | 10 | 0.0% |



## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| no_mistake | 0.29 | 1.00 | 0.45 | 32 |
| articulation | 0.50 | 0.22 | 0.31 | 9 |
| dynamics | 0.50 | 0.11 | 0.18 | 9 |
| tempo | 0.50 | 0.05 | 0.09 | 20 |
| harmony | — | 0.00 | — | 10 |
| pitch | — | 0.00 | — | 13 |
| rhythm_and_timing | 0.00 | 0.00 | — | 13 |
| technique | — | 0.00 | — | 13 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| articulation | 4 | 3.4% | 9 | 7.6% |
| dynamics | 2 | 1.7% | 9 | 7.6% |
| harmony | 0 | 0.0% | 10 | 8.4% |
| no_mistake | 109 | 91.6% | 32 | 26.9% |
| pitch | 0 | 0.0% | 13 | 10.9% |
| rhythm_and_timing | 2 | 1.7% | 13 | 10.9% |
| technique | 0 | 0.0% | 13 | 10.9% |
| tempo | 2 | 1.7% | 20 | 16.8% |


## Confusion Matrices

### Overall

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 2 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| **dynamics** | 1 | 1 | 0 | 7 | 0 | 0 | 0 | 0 |
| **harmony** | 0 | 0 | 0 | 8 | 0 | 2 | 0 | 0 |
| **no_mistake** | 0 | 0 | 0 | 32 | 0 | 0 | 0 | 0 |
| **pitch** | 0 | 0 | 0 | 13 | 0 | 0 | 0 | 0 |
| **rhythm_and_timing** | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 1 |
| **technique** | 1 | 0 | 0 | 12 | 0 | 0 | 0 | 0 |
| **tempo** | 0 | 1 | 0 | 18 | 0 | 0 | 0 | 1 |

### Single Audio

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 2 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| **dynamics** | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| **harmony** | 0 | 0 | 0 | 5 | 0 | 2 | 0 | 0 |
| **no_mistake** | 0 | 0 | 0 | 14 | 0 | 0 | 0 | 0 |
| **pitch** | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| **rhythm_and_timing** | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| **technique** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| **tempo** | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |

### Double Audio

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| **dynamics** | 1 | 1 | 0 | 5 | 0 | 0 | 0 | 0 |
| **harmony** | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| **no_mistake** | 0 | 0 | 0 | 18 | 0 | 0 | 0 | 0 |
| **pitch** | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| **rhythm_and_timing** | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 1 |
| **technique** | 1 | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
| **tempo** | 0 | 1 | 0 | 11 | 0 | 0 | 0 | 1 |

---

*Generated on 2026-03-30 12:00*
