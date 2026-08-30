# Open Ended — gemini-2.0-flash

| | |
|---|---|
| **Run ID** | `20260307_221015` |
| **Provider** | `gemini` |
| **Model** | `gemini-2.0-flash` |
| **Prompt hash** | `d867d047` |


## Summary

Overall performance on the open-ended task. Detection accuracy measures whether the model correctly identified the presence/absence of a mistake. SBERT similarity measures semantic overlap between predicted and ground truth mistake descriptions and feedback (1.0 = identical meaning, 0.0 = unrelated).

| Metric | Value |
|--------|------:|
| Total samples | 120 |
| Parse errors | 2 (1.7%) |
| Correct detections | 74 |
| **Detection accuracy** | **61.7%** |
| Precision | 0.739 |
| Recall | 0.756 |
| **F1** | **0.747** |
| SBERT mistake sim. | 0.364 |
| SBERT feedback sim. | 0.359 |
| Judge: Correct mistake | 17.9% (12/67) |
| Judge: Helpful feedback | 20.9% (14/67) |


## Per-Category Classification

How often the model predicted the correct mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| articulation | 5 | 9 | 55.6% |
| tempo | 11 | 20 | 55.0% |
| harmony | 5 | 10 | 50.0% |
| no_mistake | 9 | 32 | 28.1% |
| pitch | 3 | 13 | 23.1% |
| technique | 2 | 13 | 15.4% |
| dynamics | 1 | 9 | 11.1% |
| rhythm_and_timing | 0 | 14 | 0.0% |


## Per-Category Similarity (SBERT)

Average semantic similarity between predicted and ground truth mistake descriptions, broken down by category. Higher values indicate closer semantic match.

| Category | Samples | Mistake Sim. |
|----------|--------:|-------------:|
| articulation | 9 | 0.638 |
| tempo | 20 | 0.513 |
| harmony | 10 | 0.456 |
| dynamics | 9 | 0.424 |
| pitch | 13 | 0.322 |
| no_mistake | 32 | 0.281 |
| technique | 13 | 0.273 |
| rhythm_and_timing | 14 | 0.185 |


## LLM-as-a-Judge

A secondary LLM evaluates the quality of model predictions by comparing them against ground truth on two dimensions: mistake description (how well does the predicted mistake match?) and feedback quality (is the corrective advice useful?). Only samples where both the prediction and ground truth contain mistake content are sent to the judge.

### Sample Routing

Before reaching the judge LLM, samples are routed based on whether the prediction and ground truth contain mistake content.

- **No mistake**: neither side contains a mistake — nothing to judge.
- **False negative**: the ground truth has a mistake but the model produced none.
- **False positive**: the model reported a mistake but the ground truth has none.
- **LLM evaluated**: both sides have content — sent to the judge.

| Outcome | Count | Rate |
|---------|------:|-----:|
| No mistake | 9 | 7.5% |
| False negative | 21 | 17.5% |
| False positive | 23 | 19.2% |
| LLM evaluated | 67 | 55.8% |
| **Total** | **120** | |

### Mistake Description

How well does the predicted mistake match the ground truth?

| Rating | Count | Rate |
|--------|------:|-----:|
| correct | 12 | 17.9% |
| partial | 23 | 34.3% |
| incorrect | 32 | 47.8% |


### Feedback Quality

Is the corrective feedback specific and actionable?

| Rating | Count | Rate |
|--------|------:|-----:|
| helpful | 14 | 20.9% |
| generic | 28 | 41.8% |
| unhelpful | 25 | 37.3% |


## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| harmony | 0.71 | 0.50 | 0.59 | 10 |
| articulation | 0.56 | 0.56 | 0.56 | 9 |
| tempo | 0.24 | 0.55 | 0.33 | 20 |
| no_mistake | 0.30 | 0.28 | 0.29 | 32 |
| pitch | 0.25 | 0.23 | 0.24 | 13 |
| technique | 0.40 | 0.15 | 0.22 | 13 |
| dynamics | 0.20 | 0.11 | 0.14 | 9 |
| rhythm_and_timing | 0.00 | 0.00 | — | 14 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| articulation | 9 | 7.5% | 9 | 7.5% |
| dynamics | 5 | 4.2% | 9 | 7.5% |
| harmony | 7 | 5.8% | 10 | 8.3% |
| no_mistake | 30 | 25.0% | 32 | 26.7% |
| pitch | 12 | 10.0% | 13 | 10.8% |
| rhythm_and_timing | 6 | 5.0% | 14 | 11.7% |
| technique | 5 | 4.2% | 13 | 10.8% |
| tempo | 46 | 38.3% | 20 | 16.7% |


## Confusion Matrices

### Overall

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 5 | 1 | 0 | 1 | 1 | 0 | 0 | 1 |
| **dynamics** | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 5 |
| **harmony** | 0 | 1 | 5 | 0 | 0 | 1 | 0 | 3 |
| **no_mistake** | 4 | 2 | 1 | 9 | 4 | 1 | 0 | 11 |
| **pitch** | 0 | 0 | 1 | 3 | 3 | 1 | 2 | 3 |
| **rhythm_and_timing** | 0 | 0 | 0 | 8 | 1 | 0 | 0 | 5 |
| **technique** | 0 | 0 | 0 | 2 | 2 | 0 | 2 | 7 |
| **tempo** | 0 | 0 | 0 | 6 | 1 | 2 | 0 | 11 |

---

*Generated on 2026-03-08 21:48*
