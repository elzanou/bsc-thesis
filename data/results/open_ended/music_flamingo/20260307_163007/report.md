# Open Ended — nvidia/music-flamingo-hf

| | |
|---|---|
| **Run ID** | `20260307_163007` |
| **Provider** | `music_flamingo` |
| **Model** | `nvidia/music-flamingo-hf` |
| **Prompt hash** | `d867d047` |


## Summary

Overall performance on the open-ended task. Detection accuracy measures whether the model correctly identified the presence/absence of a mistake. SBERT similarity measures semantic overlap between predicted and ground truth mistake descriptions and feedback (1.0 = identical meaning, 0.0 = unrelated).

| Metric | Value |
|--------|------:|
| Total samples | 120 |
| Parse errors | 29 (24.2%) |
| Correct detections | 51 |
| **Detection accuracy** | **42.5%** |
| Precision | 0.714 |
| Recall | 0.672 |
| **F1** | **0.692** |
| SBERT mistake sim. | 0.295 |
| SBERT feedback sim. | 0.314 |
| Judge: Correct mistake | 21.6% (16/74) |
| Judge: Helpful feedback | 13.5% (10/74) |


## Per-Category Classification

How often the model predicted the correct mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| articulation | 7 | 9 | 77.8% |
| dynamics | 2 | 9 | 22.2% |
| no_mistake | 4 | 32 | 12.5% |
| tempo | 2 | 20 | 10.0% |
| harmony | 1 | 10 | 10.0% |
| pitch | 1 | 13 | 7.7% |
| technique | 1 | 13 | 7.7% |
| rhythm_and_timing | 0 | 14 | 0.0% |


## Per-Category Similarity (SBERT)

Average semantic similarity between predicted and ground truth mistake descriptions, broken down by category. Higher values indicate closer semantic match.

| Category | Samples | Mistake Sim. |
|----------|--------:|-------------:|
| articulation | 9 | 0.662 |
| no_mistake | 32 | 0.406 |
| dynamics | 9 | 0.280 |
| pitch | 13 | 0.255 |
| tempo | 20 | 0.230 |
| harmony | 10 | 0.226 |
| rhythm_and_timing | 14 | 0.146 |
| technique | 13 | 0.129 |


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
| No mistake | 6 | 5.0% |
| False negative | 22 | 18.3% |
| False positive | 18 | 15.0% |
| LLM evaluated | 74 | 61.7% |
| **Total** | **120** | |

### Mistake Description

How well does the predicted mistake match the ground truth?

| Rating | Count | Rate |
|--------|------:|-----:|
| correct | 16 | 21.6% |
| partial | 15 | 20.3% |
| incorrect | 43 | 58.1% |


### Feedback Quality

Is the corrective feedback specific and actionable?

| Rating | Count | Rate |
|--------|------:|-----:|
| helpful | 10 | 13.5% |
| generic | 21 | 28.4% |
| unhelpful | 43 | 58.1% |


## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| articulation | 0.50 | 0.78 | 0.61 | 9 |
| dynamics | 0.40 | 0.40 | 0.40 | 5 |
| harmony | 0.33 | 0.12 | 0.18 | 8 |
| tempo | 0.25 | 0.14 | 0.18 | 14 |
| no_mistake | 0.15 | 0.21 | 0.18 | 19 |
| pitch | 0.25 | 0.12 | 0.17 | 8 |
| technique | 0.17 | 0.11 | 0.13 | 9 |
| dynamics, articulation | 0.00 | — | — | 0 |
| rhythm | 0.00 | — | — | 0 |
| rhythm.and timing | 0.00 | — | — | 0 |
| rhythm.andtiming | 0.00 | — | — | 0 |
| rhythm.entiming | 0.00 | — | — | 0 |
| rhythm[outro] | 0.00 | — | — | 0 |
| rhythm_and_timing | — | 0.00 | — | 10 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| articulation | 14 | 17.1% | 9 | 11.0% |
| dynamics | 5 | 6.1% | 5 | 6.1% |
| dynamics, articulation | 1 | 1.2% | 0 | 0.0% |
| harmony | 3 | 3.7% | 8 | 9.8% |
| no_mistake | 26 | 31.7% | 19 | 23.2% |
| pitch | 4 | 4.9% | 8 | 9.8% |
| rhythm | 5 | 6.1% | 0 | 0.0% |
| rhythm.and timing | 5 | 6.1% | 0 | 0.0% |
| rhythm.andtiming | 1 | 1.2% | 0 | 0.0% |
| rhythm.entiming | 3 | 3.7% | 0 | 0.0% |
| rhythm[outro] | 1 | 1.2% | 0 | 0.0% |
| rhythm_and_timing | 0 | 0.0% | 10 | 12.2% |
| technique | 6 | 7.3% | 9 | 11.0% |
| tempo | 8 | 9.8% | 14 | 17.1% |


## Confusion Matrices

### Overall

| Actual \ Predicted | articulation | dynamics | dynamics, articulation | harmony | no_mistake | pitch | rhythm | rhythm.and timing | rhythm.andtiming | rhythm.entiming | rhythm[outro] | rhythm_and_timing | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 7 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **dynamics** | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **dynamics, articulation** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **harmony** | 0 | 0 | 0 | 1 | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| **no_mistake** | 3 | 1 | 0 | 0 | 4 | 1 | 3 | 1 | 1 | 2 | 1 | 0 | 0 | 2 |
| **pitch** | 0 | 0 | 0 | 2 | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **rhythm** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **rhythm.and timing** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **rhythm.andtiming** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **rhythm.entiming** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **rhythm[outro]** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **rhythm_and_timing** | 2 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |
| **technique** | 0 | 1 | 0 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| **tempo** | 2 | 1 | 1 | 0 | 2 | 0 | 0 | 3 | 0 | 1 | 0 | 0 | 2 | 2 |

---

*Generated on 2026-03-08 21:49*
