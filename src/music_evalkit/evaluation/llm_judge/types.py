from enum import StrEnum

from pydantic import BaseModel


class QualityCategory(StrEnum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


class FeedbackCategory(StrEnum):
    HELPFUL = "helpful"
    GENERIC = "generic"
    UNHELPFUL = "unhelpful"


class JudgeSource(StrEnum):
    """How a judge result was produced.

    The LLM judge can only run when both the prediction and ground truth
    contain mistake + feedback text to compare. When either side is empty,
    the outcome is a detection result (no LLM call).
    """

    LLM = "llm"
    NO_MISTAKE = "no_mistake"        # neither side has mistake text
    MISSED = "missed"                # model produced no mistake text, GT has one
    HALLUCINATED = "hallucinated"    # model produced mistake text, GT has none


class JudgeResult(BaseModel, frozen=True):
    """Result from LLM judge evaluation.

    Each sample is classified on two dimensions:
    - Mistake Description: correct / partially_correct / incorrect
    - Feedback Quality: helpful / generic / unhelpful

    When source is not LLM, all categories are None — one or both sides
    lacked mistake content, so there was nothing for the judge to compare.
    """

    mistake: QualityCategory | None
    mistake_reasoning: str
    feedback: FeedbackCategory | None
    feedback_reasoning: str
    raw_response: str
    source: JudgeSource


class JudgeMetrics(BaseModel, frozen=True):
    """Aggregated metrics from LLM-as-a-Judge evaluation.

    Tracks detection outcomes separately from quality assessments.
    All rates are normalized: routing rates use total as denominator,
    quality rates use judged as denominator.
    """

    total: int

    # Detection outcomes — one or both sides lacked mistake content
    no_mistake: int = 0       # neither side had a mistake
    missed: int = 0           # model produced nothing, GT had a mistake
    hallucinated: int = 0     # model produced a mistake, GT had none

    # Quality assessment (both sides had content → sent to LLM)
    llm_evaluated: int = 0
    judged: int = 0
    parse_errors: int = 0

    # Category distributions (only for successfully judged samples)
    mistake_distribution: dict[str, int] = {}
    feedback_distribution: dict[str, int] = {}

    # Normalized rates
    no_mistake_rate: float = 0.0
    missed_rate: float = 0.0
    hallucinated_rate: float = 0.0
    llm_evaluated_rate: float = 0.0
    parse_error_rate: float = 0.0

    # Quality rates (% of judged)
    mistake_rates: dict[str, float] = {}
    feedback_rates: dict[str, float] = {}
