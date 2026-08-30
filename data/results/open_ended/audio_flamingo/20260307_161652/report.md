# Open Ended — nvidia/audio-flamingo-3-hf

| | |
|---|---|
| **Run ID** | `20260307_161652` |
| **Provider** | `audio_flamingo` |
| **Model** | `nvidia/audio-flamingo-3-hf` |
| **Prompt hash** | `d867d047` |


## Summary

Overall performance on the open-ended task. Detection accuracy measures whether the model correctly identified the presence/absence of a mistake. SBERT similarity measures semantic overlap between predicted and ground truth mistake descriptions and feedback (1.0 = identical meaning, 0.0 = unrelated).

| Metric | Value |
|--------|------:|
| Total samples | 120 |
| Parse errors | 22 (18.3%) |
| Correct detections | 59 |
| **Detection accuracy** | **49.2%** |
| Precision | 0.757 |
| Recall | 0.707 |
| **F1** | **0.731** |
| SBERT mistake sim. | 0.279 |
| SBERT feedback sim. | 0.307 |
| Judge: Correct mistake | 8.8% (6/68) |
| Judge: Helpful feedback | 20.6% (14/68) |


## Per-Category Classification

How often the model predicted the correct mistake category (e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| articulation | 2 | 9 | 22.2% |
| rhythm_and_timing | 2 | 14 | 14.3% |
| no_mistake | 4 | 32 | 12.5% |
| tempo | 2 | 20 | 10.0% |
| harmony | 1 | 10 | 10.0% |
| pitch | 1 | 13 | 7.7% |
| technique | 1 | 13 | 7.7% |
| dynamics | 0 | 9 | 0.0% |


## Per-Category Similarity (SBERT)

Average semantic similarity between predicted and ground truth mistake descriptions, broken down by category. Higher values indicate closer semantic match.

| Category | Samples | Mistake Sim. |
|----------|--------:|-------------:|
| harmony | 10 | 0.333 |
| pitch | 13 | 0.322 |
| rhythm_and_timing | 14 | 0.318 |
| no_mistake | 32 | 0.312 |
| tempo | 20 | 0.304 |
| articulation | 9 | 0.299 |
| technique | 13 | 0.132 |
| dynamics | 9 | 0.108 |


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
| False positive | 17 | 14.2% |
| LLM evaluated | 75 | 62.5% |
| **Total** | **120** | |

*7 of 75 LLM-evaluated samples had parse errors and are excluded from quality ratings.*


### Mistake Description

How well does the predicted mistake match the ground truth?

| Rating | Count | Rate |
|--------|------:|-----:|
| correct | 6 | 8.8% |
| partial | 12 | 17.6% |
| incorrect | 50 | 73.5% |


### Feedback Quality

Is the corrective feedback specific and actionable?

| Rating | Count | Rate |
|--------|------:|-----:|
| helpful | 14 | 20.6% |
| generic | 6 | 8.8% |
| unhelpful | 48 | 70.6% |


## Per-Class Precision / Recall / F1

Per-class metrics computed from the confusion matrix. Precision = how often a predicted class is correct. Recall = how often an actual class is detected. Support = number of ground truth samples per class.

| Category | P | R | F1 | Support |
|----------|--:|--:|---:|--------:|
| articulation | 0.29 | 0.33 | 0.31 | 6 |
| rhythm_and_timing | 0.29 | 0.29 | 0.29 | 7 |
| no_mistake | 0.25 | 0.31 | 0.28 | 13 |
| harmony | 0.33 | 0.17 | 0.22 | 6 |
| tempo | 0.33 | 0.15 | 0.21 | 13 |
| technique | 0.17 | 0.20 | 0.18 | 5 |
| pitch | 0.33 | 0.11 | 0.17 | 9 |
| dynamics | 0.00 | 0.00 | — | 3 |
| no_mmistake | 0.00 | — | — | 0 |
| no_mιστάκε | 0.00 | — | — | 0 |
| pitch, rhythm_andtiming, tempo | 0.00 | — | — | 0 |
| pitch, technique | 0.00 | — | — | 0 |
| pitch, technique, articulation | 0.00 | — | — | 0 |
| rhythm-and-timing | 0.00 | — | — | 0 |
| rhythm_andtiming | 0.00 | — | — | 0 |


## Prediction vs Ground Truth Distribution

How often each category was predicted vs how often it actually appears. Large discrepancies indicate systematic bias (e.g., over-predicting a category).

| Category | Predicted | (%) | Actual | (%) |
|----------|----------:|----:|-------:|----:|
| articulation | 7 | 11.3% | 6 | 9.7% |
| dynamics | 6 | 9.7% | 3 | 4.8% |
| harmony | 3 | 4.8% | 6 | 9.7% |
| no_mistake | 16 | 25.8% | 13 | 21.0% |
| no_mmistake | 1 | 1.6% | 0 | 0.0% |
| no_mιστάκε | 1 | 1.6% | 0 | 0.0% |
| pitch | 3 | 4.8% | 9 | 14.5% |
| pitch, rhythm_andtiming, tempo | 1 | 1.6% | 0 | 0.0% |
| pitch, technique | 1 | 1.6% | 0 | 0.0% |
| pitch, technique, articulation | 1 | 1.6% | 0 | 0.0% |
| rhythm-and-timing | 1 | 1.6% | 0 | 0.0% |
| rhythm_and_timing | 7 | 11.3% | 7 | 11.3% |
| rhythm_andtiming | 2 | 3.2% | 0 | 0.0% |
| technique | 6 | 9.7% | 5 | 8.1% |
| tempo | 6 | 9.7% | 13 | 21.0% |


## Confusion Matrices

### Overall

| Actual \ Predicted | articulation | dynamics | harmony | no_mistake | no_mmistake | no_mιστάκε | pitch | pitch, rhythm_andtiming, tempo | pitch, technique | pitch, technique, articulation | rhythm-and-timing | rhythm_and_timing | rhythm_andtiming | technique | tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **articulation** | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| **dynamics** | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **harmony** | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| **no_mistake** | 3 | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 2 |
| **no_mmistake** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **no_mιστάκε** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **pitch** | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 4 | 1 |
| **pitch, rhythm_andtiming, tempo** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **pitch, technique** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **pitch, technique, articulation** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **rhythm-and-timing** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **rhythm_and_timing** | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 1 |
| **rhythm_andtiming** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **technique** | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| **tempo** | 0 | 2 | 2 | 3 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |

---

*Generated on 2026-03-09 14:46*
