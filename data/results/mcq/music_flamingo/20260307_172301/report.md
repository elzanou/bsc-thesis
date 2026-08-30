# Mcq — nvidia/music-flamingo-hf

| | |
|---|---|
| **Run ID** | `20260307_172301` |
| **Provider** | `music_flamingo` |
| **Model** | `nvidia/music-flamingo-hf` |
| **Prompt hash** | `ac297321` |


## Summary

Overall performance on the MCQ task. Accuracy measures exact-match between predicted and ground truth answers. Precision, recall, and F1 are macro-averaged across all classes.

| Metric | Value |
|--------|------:|
| Total samples | 120 |
| Parse errors | 0 (0.0%) |
| Correct | 34 |
| **Accuracy** | **28.3%** |
| Precision (macro) | 0.254 |
| Recall (macro) | 0.211 |
| **F1 (macro)** | **0.205** |


## Results by Audio Setting

Single-audio samples provide only the student recording. Double-audio samples include both a reference performance and the student recording, separated by a beep.

| Setting | Samples | Accuracy |
|---------|--------:|---------:|
| Single audio | 48 | 35.4% |
| Double audio | 72 | 23.6% |


## Per-Category Accuracy

Accuracy broken down by mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| no_mistake | 21 | 32 | 65.6% |
| articulation | 3 | 9 | 33.3% |
| tempo | 5 | 20 | 25.0% |
| dynamics | 2 | 9 | 22.2% |
| pitch | 2 | 13 | 15.4% |
| rhythm_and_timing | 1 | 14 | 7.1% |
| technique | 0 | 13 | 0.0% |
| harmony | 0 | 10 | 0.0% |



## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| articulation | 0.50 | 0.33 | 0.40 | 9 |
| no_mistake | 0.28 | 0.66 | 0.39 | 32 |
| tempo | 0.45 | 0.25 | 0.32 | 20 |
| pitch | 0.50 | 0.15 | 0.24 | 13 |
| dynamics | 0.20 | 0.22 | 0.21 | 9 |
| rhythm_and_timing | 0.10 | 0.07 | 0.08 | 14 |
| harmony | 0.00 | 0.00 | — | 10 |
| technique | 0.00 | 0.00 | — | 13 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| articulation | 6 | 5.0% | 9 | 7.5% |
| dynamics | 10 | 8.3% | 9 | 7.5% |
| harmony | 2 | 1.7% | 10 | 8.3% |
| no_mistake | 76 | 63.3% | 32 | 26.7% |
| pitch | 4 | 3.3% | 13 | 10.8% |
| rhythm_and_timing | 10 | 8.3% | 14 | 11.7% |
| technique | 1 | 0.8% | 13 | 10.8% |
| tempo | 11 | 9.2% | 20 | 16.7% |


## Confusion Matrices

### Overall

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 3 | 2 | 0 | 4 | 0 | 0 | 0 | 0 |
| **dynamics** | 2 | 2 | 0 | 4 | 0 | 0 | 0 | 1 |
| **harmony** | 0 | 0 | 0 | 7 | 0 | 3 | 0 | 0 |
| **no_mistake** | 0 | 4 | 0 | 21 | 2 | 0 | 1 | 4 |
| **pitch** | 0 | 0 | 2 | 7 | 2 | 2 | 0 | 0 |
| **rhythm_and_timing** | 0 | 0 | 0 | 12 | 0 | 1 | 0 | 1 |
| **technique** | 1 | 1 | 0 | 11 | 0 | 0 | 0 | 0 |
| **tempo** | 0 | 1 | 0 | 10 | 0 | 4 | 0 | 5 |

### Single Audio

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 1 | 1 | 0 | 4 | 0 | 0 | 0 | 0 |
| **dynamics** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **harmony** | 0 | 0 | 0 | 4 | 0 | 3 | 0 | 0 |
| **no_mistake** | 0 | 1 | 0 | 11 | 1 | 0 | 0 | 1 |
| **pitch** | 0 | 0 | 1 | 3 | 1 | 0 | 0 | 0 |
| **rhythm_and_timing** | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| **technique** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| **tempo** | 0 | 0 | 0 | 2 | 0 | 2 | 0 | 3 |

### Double Audio

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **dynamics** | 1 | 1 | 0 | 4 | 0 | 0 | 0 | 1 |
| **harmony** | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| **no_mistake** | 0 | 3 | 0 | 10 | 1 | 0 | 1 | 3 |
| **pitch** | 0 | 0 | 1 | 4 | 1 | 2 | 0 | 0 |
| **rhythm_and_timing** | 0 | 0 | 0 | 6 | 0 | 1 | 0 | 1 |
| **technique** | 1 | 1 | 0 | 10 | 0 | 0 | 0 | 0 |
| **tempo** | 0 | 1 | 0 | 8 | 0 | 2 | 0 | 2 |

---

*Generated on 2026-03-30 12:00*
