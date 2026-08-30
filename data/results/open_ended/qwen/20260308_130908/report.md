# Open Ended — qwen2.5-omni-7b

| | |
|---|---|
| **Run ID** | `20260308_130908` |
| **Provider** | `qwen` |
| **Model** | `qwen2.5-omni-7b` |
| **Prompt hash** | `d867d047` |


## Summary

Overall performance on the open-ended task. Detection accuracy measures whether the model correctly identified the presence/absence of a mistake. SBERT similarity measures semantic overlap between predicted and ground truth mistake descriptions and feedback (1.0 = identical meaning, 0.0 = unrelated).

| Metric | Value |
|--------|------:|
| Total samples | 120 |
| Parse errors | 1 (0.8%) |
| Correct detections | 39 |
| **Detection accuracy** | **32.5%** |
| Precision | 0.667 |
| Recall | 0.161 |
| **F1** | **0.259** |
| SBERT mistake sim. | 0.267 |
| SBERT feedback sim. | 0.276 |
| Judge: Correct mistake | 40.0% (6/15) |
| Judge: Helpful feedback | 40.0% (6/15) |


## Per-Category Classification

How often the model predicted the correct mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| no_mistake | 25 | 32 | 78.1% |
| articulation | 5 | 9 | 55.6% |
| dynamics | 2 | 9 | 22.2% |
| pitch | 0 | 13 | 0.0% |
| tempo | 0 | 20 | 0.0% |
| rhythm_and_timing | 0 | 14 | 0.0% |
| technique | 0 | 13 | 0.0% |
| harmony | 0 | 10 | 0.0% |


## Per-Category Similarity (SBERT)

Average semantic similarity between predicted and ground truth mistake descriptions, broken down by category. Higher values indicate closer semantic match.

| Category | Samples | Mistake Sim. |
|----------|--------:|-------------:|
| no_mistake | 32 | 0.781 |
| articulation | 9 | 0.360 |
| dynamics | 9 | 0.165 |
| pitch | 13 | 0.075 |
| harmony | 10 | 0.045 |
| tempo | 20 | 0.022 |
| rhythm_and_timing | 14 | 0.018 |
| technique | 13 | 0.016 |


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
| No mistake | 25 | 20.8% |
| False negative | 73 | 60.8% |
| False positive | 7 | 5.8% |
| LLM evaluated | 15 | 12.5% |
| **Total** | **120** | |

### Mistake Description

How well does the predicted mistake match the ground truth?

| Rating | Count | Rate |
|--------|------:|-----:|
| correct | 6 | 40.0% |
| partial | 3 | 20.0% |
| incorrect | 6 | 40.0% |


### Feedback Quality

Is the corrective feedback specific and actionable?

| Rating | Count | Rate |
|--------|------:|-----:|
| helpful | 6 | 40.0% |
| generic | 6 | 40.0% |
| unhelpful | 3 | 20.0% |


## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| articulation | 0.45 | 0.56 | 0.50 | 9 |
| no_mistake | 0.26 | 0.78 | 0.38 | 32 |
| dynamics | 0.67 | 0.22 | 0.33 | 9 |
| harmony | — | 0.00 | — | 10 |
| pitch | — | 0.00 | — | 13 |
| rhythm_and_timing | 0.00 | 0.00 | — | 13 |
| technique | 0.00 | 0.00 | — | 13 |
| tempo | 0.00 | 0.00 | — | 20 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| articulation | 11 | 9.2% | 9 | 7.6% |
| dynamics | 3 | 2.5% | 9 | 7.6% |
| harmony | 0 | 0.0% | 10 | 8.4% |
| no_mistake | 98 | 82.4% | 32 | 26.9% |
| pitch | 0 | 0.0% | 13 | 10.9% |
| rhythm_and_timing | 1 | 0.8% | 13 | 10.9% |
| technique | 1 | 0.8% | 13 | 10.9% |
| tempo | 5 | 4.2% | 20 | 16.8% |


## Confusion Matrices

### Overall

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | pitch | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 5 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| **dynamics** | 0 | 2 | 0 | 7 | 0 | 0 | 0 | 0 |
| **harmony** | 0 | 0 | 0 | 9 | 0 | 1 | 0 | 0 |
| **no_mistake** | 5 | 1 | 0 | 25 | 0 | 0 | 0 | 1 |
| **pitch** | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 3 |
| **rhythm_and_timing** | 1 | 0 | 0 | 12 | 0 | 0 | 0 | 0 |
| **technique** | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 1 |
| **tempo** | 0 | 0 | 0 | 19 | 0 | 0 | 1 | 0 |

---

*Generated on 2026-03-08 21:49*
