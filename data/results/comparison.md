# Model Comparison

Comparing **4 models** across MCQ, pairwise, and open-ended tasks. Best result per metric is shown in **bold**.

## Main Results

| Task | Metric | gemini-2.0-flash | audio-flamingo-3-hf | music-flamingo-hf | qwen2.5-omni-7b |
|------|--------|---:|---:|---:|---:|
| **Mcq** | Accuracy ↑ | **0.433** | 0.333 | 0.283 | 0.300 |
|  | F1 (macro) ↑ | **0.453** | 0.326 | 0.205 | 0.129 |
|  | Parse errors | 0 | 1 | 0 | 1 |
| **Pairwise** | Accuracy ↑ | **0.683** | 0.433 | 0.367 | 0.467 |
|  | F1 (macro) ↑ | **0.683** | 0.481 | 0.355 | 0.443 |
|  | Parse errors | 0 | 8 | 0 | 1 |
| **Open Ended** | Det. accuracy ↑ | **0.617** | 0.492 | 0.425 | 0.325 |
|  | F1 ↑ | **0.747** | 0.731 | 0.692 | 0.259 |
|  | SBERT mistake ↑ | **0.364** | 0.279 | 0.295 | 0.267 |
|  | SBERT feedback ↑ | **0.359** | 0.307 | 0.314 | 0.276 |
|  | Parse errors | 2 | 22 | 29 | 1 |


## Accuracy by Audio Setting

Single-audio samples provide only the student recording. Double-audio samples include both a reference and student recording.

| Task | Setting | gemini-2.0-flash | audio-flamingo-3-hf | music-flamingo-hf | qwen2.5-omni-7b |
|------|---------|---:|---:|---:|---:|
| **Mcq** | Single | **0.396** | 0.354 | 0.354 | 0.333 |
|  | Double | **0.458** | 0.324 | 0.236 | 0.282 |


## LLM-as-a-Judge (Open-Ended)

Quality assessment by a secondary LLM. Samples are first routed based on content presence before reaching the judge:

- **No mistake**: neither the model nor the ground truth contains a mistake — nothing to judge.
- **False negative**: the ground truth has a mistake but the model produced none.
- **False positive**: the model reported a mistake but the ground truth has none.
- **LLM evaluated**: both sides have content — sent to the judge for quality rating.

Routing rates are normalized by total samples. Quality rates are normalized by the number of successfully judged samples.

| Metric | gemini-2.0-flash | audio-flamingo-3-hf | music-flamingo-hf | qwen2.5-omni-7b |
|--------|---:|---:|---:|---:|
| No mistake rate | 7.5% | 7.5% | 10.8% | 20.8% |
| False negative rate ↓ | **17.5%** | 21.7% | 31.7% | 60.8% |
| False positive rate ↓ | 19.2% | 18.3% | 15.8% | **5.8%** |
| LLM evaluated rate | 55.8% | 52.5% | 41.7% | 12.5% |
| Judged | 67 | 62 | 50 | 15 |
| Parse errors | 0 | 1 | 0 | 0 |
| | — | — | — | — |
| Correct mistake ↑ | 22.4% | 8.1% | 14.0% | **46.7%** |
| Helpful feedback ↑ | 23.9% | 12.9% | 16.0% | **40.0%** |


## Per-Category Accuracy — Mcq

| Category | gemini-2.0-flash | audio-flamingo-3-hf | music-flamingo-hf | qwen2.5-omni-7b |
|----------|---:|---:|---:|---:|
| no_mistake | 0.531 | 0.500 | 0.656 | **1.000** |
| articulation | 0.556 | **0.667** | 0.333 | 0.222 |
| dynamics | **0.556** | 0.333 | 0.222 | 0.111 |
| technique | **0.615** | 0.385 | 0.000 | 0.000 |
| pitch | **0.385** | 0.385 | 0.154 | 0.000 |
| tempo | **0.350** | 0.050 | 0.250 | 0.050 |
| rhythm_and_timing | **0.286** | 0.214 | 0.071 | 0.000 |
| harmony | **0.100** | 0.100 | 0.000 | 0.000 |


## Per-Category Accuracy — Pairwise

| Category | gemini-2.0-flash | audio-flamingo-3-hf | music-flamingo-hf | qwen2.5-omni-7b |
|----------|---:|---:|---:|---:|
| tempo | **0.800** | 0.467 | 0.467 | 0.533 |
| articulation | **1.000** | 0.375 | 0.375 | 0.500 |
| harmony | 0.400 | 0.600 | 0.200 | **1.000** |
| technique | **0.750** | 0.625 | 0.125 | 0.500 |
| dynamics | **0.778** | 0.556 | 0.333 | 0.333 |
| pitch | **0.429** | 0.286 | 0.429 | 0.286 |
| rhythm_and_timing | 0.375 | 0.125 | **0.500** | 0.250 |


## Per-Category Accuracy — Open Ended

| Category | gemini-2.0-flash | audio-flamingo-3-hf | music-flamingo-hf | qwen2.5-omni-7b |
|----------|---:|---:|---:|---:|
| articulation | 0.556 | 0.222 | **0.778** | 0.556 |
| no_mistake | 0.281 | 0.125 | 0.125 | **0.781** |
| tempo | **0.550** | 0.100 | 0.100 | 0.000 |
| harmony | **0.500** | 0.100 | 0.100 | 0.000 |
| dynamics | 0.111 | 0.000 | **0.222** | 0.222 |
| pitch | **0.231** | 0.077 | 0.077 | 0.000 |
| technique | **0.154** | 0.077 | 0.077 | 0.000 |
| rhythm_and_timing | 0.000 | **0.143** | 0.000 | 0.000 |

