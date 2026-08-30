"""The judge only runs when both sides have mistake content; otherwise the
sample is recorded as a detection outcome (no_mistake/missed/hallucinated)
without calling the LLM.
"""

from music_evalkit.evaluation.llm_judge.helpers import try_detect
from music_evalkit.evaluation.llm_judge.judge import (
    PERMANENT_ERRORS,
    LLMJudge,
)
from music_evalkit.evaluation.llm_judge.metrics import compute_judge_metrics
from music_evalkit.evaluation.llm_judge.runner import run_judge
from music_evalkit.evaluation.llm_judge.types import (
    FeedbackCategory,
    JudgeMetrics,
    JudgeResult,
    JudgeSource,
    QualityCategory,
)

__all__ = [
    "LLMJudge",
    "JudgeResult",
    "JudgeMetrics",
    "JudgeSource",
    "QualityCategory",
    "FeedbackCategory",
    "run_judge",
    "compute_judge_metrics",
    "try_detect",
    "PERMANENT_ERRORS",
]
