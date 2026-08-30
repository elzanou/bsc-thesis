# Mcq — gemini-2.0-flash

| | |
|---|---|
| **Run ID** | `20260307_215631` |
| **Provider** | `gemini` |
| **Model** | `gemini-2.0-flash` |
| **Prompt hash** | `ac297321` |


## Summary

Overall performance on the MCQ task. Accuracy measures exact-match between predicted and ground truth answers. Precision, recall, and F1 are macro-averaged across all classes.

| Metric | Value |
|--------|------:|
| Total samples | 120 |
| Parse errors | 0 (0.0%) |
| Correct | 52 |
| **Accuracy** | **43.3%** |
| Precision (macro) | 0.545 |
| Recall (macro) | 0.422 |
| **F1 (macro)** | **0.453** |


## Results by Audio Setting

Single-audio samples provide only the student recording. Double-audio samples include both a reference performance and the student recording, separated by a beep.

| Setting | Samples | Accuracy |
|---------|--------:|---------:|
| Single audio | 48 | 39.6% |
| Double audio | 72 | 45.8% |


## Per-Category Accuracy

Accuracy broken down by mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| technique | 8 | 13 | 61.5% |
| dynamics | 5 | 9 | 55.6% |
| articulation | 5 | 9 | 55.6% |
| no_mistake | 17 | 32 | 53.1% |
| pitch | 5 | 13 | 38.5% |
| tempo | 7 | 20 | 35.0% |
| rhythm_and_timing | 4 | 14 | 28.6% |
| harmony | 1 | 10 | 10.0% |



## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| articulation | 1.00 | 0.56 | 0.71 | 9 |
| technique | 0.73 | 0.62 | 0.67 | 13 |
| dynamics | 0.62 | 0.56 | 0.59 | 9 |
| tempo | 0.50 | 0.35 | 0.41 | 20 |
| no_mistake | 0.31 | 0.53 | 0.40 | 32 |
| pitch | 0.36 | 0.38 | 0.37 | 13 |
| rhythm_and_timing | 0.33 | 0.29 | 0.31 | 14 |
| harmony | 0.50 | 0.10 | 0.17 | 10 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| articulation | 5 | 4.2% | 9 | 7.5% |
| dynamics | 8 | 6.7% | 9 | 7.5% |
| harmony | 2 | 1.7% | 10 | 8.3% |
| no_mistake | 54 | 45.0% | 32 | 26.7% |
| pitch | 14 | 11.7% | 13 | 10.8% |
| rhythm_and_timing | 12 | 10.0% | 14 | 11.7% |
| technique | 11 | 9.2% | 13 | 10.8% |
| tempo | 14 | 11.7% | 20 | 16.7% |


## Confusion Matrices

### Overall

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 5 | 0 | 0 | 3 | 0 | 1 | 0 | 0 |
| **dynamics** | 0 | 5 | 0 | 2 | 0 | 0 | 0 | 2 |
| **harmony** | 0 | 0 | 1 | 4 | 2 | 3 | 0 | 0 |
| **no_mistake** | 0 | 3 | 0 | 17 | 5 | 0 | 3 | 4 |
| **pitch** | 0 | 0 | 1 | 7 | 5 | 0 | 0 | 0 |
| **rhythm_and_timing** | 0 | 0 | 0 | 7 | 2 | 4 | 0 | 1 |
| **technique** | 0 | 0 | 0 | 5 | 0 | 0 | 8 | 0 |
| **tempo** | 0 | 0 | 0 | 9 | 0 | 4 | 0 | 7 |

### Single Audio

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| **dynamics** | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| **harmony** | 0 | 0 | 1 | 4 | 0 | 2 | 0 | 0 |
| **no_mistake** | 0 | 2 | 0 | 11 | 0 | 0 | 0 | 1 |
| **pitch** | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| **rhythm_and_timing** | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 1 |
| **technique** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| **tempo** | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 3 |

### Double Audio

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| **dynamics** | 0 | 4 | 0 | 1 | 0 | 0 | 0 | 2 |
| **harmony** | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 |
| **no_mistake** | 0 | 1 | 0 | 6 | 5 | 0 | 3 | 3 |
| **pitch** | 0 | 0 | 1 | 2 | 5 | 0 | 0 | 0 |
| **rhythm_and_timing** | 0 | 0 | 0 | 2 | 2 | 4 | 0 | 0 |
| **technique** | 0 | 0 | 0 | 4 | 0 | 0 | 8 | 0 |
| **tempo** | 0 | 0 | 0 | 5 | 0 | 4 | 0 | 4 |

---

*Generated on 2026-03-30 11:59*
