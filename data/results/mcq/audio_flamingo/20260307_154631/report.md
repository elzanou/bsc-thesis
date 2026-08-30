# Mcq — nvidia/audio-flamingo-3-hf

| | |
|---|---|
| **Run ID** | `20260307_154631` |
| **Provider** | `audio_flamingo` |
| **Model** | `nvidia/audio-flamingo-3-hf` |
| **Prompt hash** | `ac297321` |


## Summary

Overall performance on the MCQ task. Accuracy measures exact-match between predicted and ground truth answers. Precision, recall, and F1 are macro-averaged across all classes.

| Metric | Value |
|--------|------:|
| Total samples | 120 |
| Parse errors | 1 (0.8%) |
| Correct | 40 |
| **Accuracy** | **33.3%** |
| Precision (macro) | 0.405 |
| Recall (macro) | 0.334 |
| **F1 (macro)** | **0.326** |


## Results by Audio Setting

Single-audio samples provide only the student recording. Double-audio samples include both a reference performance and the student recording, separated by a beep.

| Setting | Samples | Accuracy |
|---------|--------:|---------:|
| Single audio | 48 | 35.4% |
| Double audio | 71 | 32.4% |


## Per-Category Accuracy

Accuracy broken down by mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| articulation | 6 | 9 | 66.7% |
| no_mistake | 16 | 32 | 50.0% |
| pitch | 5 | 13 | 38.5% |
| technique | 5 | 13 | 38.5% |
| dynamics | 3 | 9 | 33.3% |
| rhythm_and_timing | 3 | 14 | 21.4% |
| harmony | 1 | 10 | 10.0% |
| tempo | 1 | 20 | 5.0% |



## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| articulation | 0.50 | 0.67 | 0.57 | 9 |
| technique | 1.00 | 0.38 | 0.56 | 13 |
| no_mistake | 0.33 | 0.50 | 0.40 | 32 |
| pitch | 0.33 | 0.38 | 0.36 | 13 |
| dynamics | 0.25 | 0.38 | 0.30 | 8 |
| rhythm_and_timing | 0.17 | 0.21 | 0.19 | 14 |
| harmony | 0.50 | 0.10 | 0.17 | 10 |
| tempo | 0.17 | 0.05 | 0.08 | 20 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| articulation | 12 | 10.1% | 9 | 7.6% |
| dynamics | 12 | 10.1% | 8 | 6.7% |
| harmony | 2 | 1.7% | 10 | 8.4% |
| no_mistake | 49 | 41.2% | 32 | 26.9% |
| pitch | 15 | 12.6% | 13 | 10.9% |
| rhythm_and_timing | 18 | 15.1% | 14 | 11.8% |
| technique | 5 | 4.2% | 13 | 10.9% |
| tempo | 6 | 5.0% | 20 | 16.8% |


## Confusion Matrices

### Overall

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 6 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| **dynamics** | 1 | 3 | 0 | 4 | 0 | 0 | 0 | 0 |
| **harmony** | 0 | 0 | 1 | 4 | 0 | 5 | 0 | 0 |
| **no_mistake** | 0 | 5 | 0 | 16 | 6 | 0 | 0 | 5 |
| **pitch** | 0 | 0 | 1 | 4 | 5 | 3 | 0 | 0 |
| **rhythm_and_timing** | 1 | 0 | 0 | 6 | 4 | 3 | 0 | 0 |
| **technique** | 4 | 2 | 0 | 2 | 0 | 0 | 5 | 0 |
| **tempo** | 0 | 2 | 0 | 10 | 0 | 7 | 0 | 1 |

### Single Audio

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 4 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| **dynamics** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **harmony** | 0 | 0 | 1 | 2 | 0 | 4 | 0 | 0 |
| **no_mistake** | 0 | 3 | 0 | 7 | 3 | 0 | 0 | 1 |
| **pitch** | 0 | 0 | 0 | 1 | 2 | 2 | 0 | 0 |
| **rhythm_and_timing** | 0 | 0 | 0 | 3 | 2 | 1 | 0 | 0 |
| **technique** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **tempo** | 0 | 0 | 0 | 2 | 0 | 4 | 0 | 1 |

### Double Audio

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| **dynamics** | 0 | 2 | 0 | 4 | 0 | 0 | 0 | 0 |
| **harmony** | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 |
| **no_mistake** | 0 | 2 | 0 | 9 | 3 | 0 | 0 | 4 |
| **pitch** | 0 | 0 | 1 | 3 | 3 | 1 | 0 | 0 |
| **rhythm_and_timing** | 1 | 0 | 0 | 3 | 2 | 2 | 0 | 0 |
| **technique** | 3 | 2 | 0 | 2 | 0 | 0 | 5 | 0 |
| **tempo** | 0 | 2 | 0 | 8 | 0 | 3 | 0 | 0 |

---

*Generated on 2026-03-30 11:59*
