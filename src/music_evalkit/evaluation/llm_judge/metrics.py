import sys

from music_evalkit.evaluation.llm_judge.types import (
    FeedbackCategory,
    JudgeMetrics,
    JudgeSource,
    QualityCategory,
)


def compute_judge_metrics(judge_results: list[dict]) -> JudgeMetrics:
    """Compute aggregated judge metrics from per-sample results.

    Args:
        judge_results: List of dicts, each with at least 'source',
            'mistake', 'feedback' keys.

    Returns:
        JudgeMetrics with detection breakdown and quality distributions.
    """
    total = len(judge_results)

    no_mistake = sum(
        1 for r in judge_results if r.get("source") == JudgeSource.NO_MISTAKE
    )
    missed = sum(
        1 for r in judge_results if r.get("source") == JudgeSource.MISSED
    )
    hallucinated = sum(
        1 for r in judge_results if r.get("source") == JudgeSource.HALLUCINATED
    )

    llm_results = [
        r for r in judge_results if r.get("source") == JudgeSource.LLM
    ]

    mistake_cats = [
        r["mistake"] for r in llm_results
        if r.get("mistake") is not None
    ]
    feedback_cats = [
        r["feedback"] for r in llm_results
        if r.get("feedback") is not None
    ]

    judged = min(len(mistake_cats), len(feedback_cats)) if llm_results else 0
    parse_errors = len(llm_results) - len(mistake_cats)

    mistake_dist = {c.value: 0 for c in QualityCategory}
    feedback_dist = {c.value: 0 for c in FeedbackCategory}

    for c in mistake_cats:
        mistake_dist[c] = mistake_dist.get(c, 0) + 1
    for c in feedback_cats:
        feedback_dist[c] = feedback_dist.get(c, 0) + 1

    if parse_errors > 0 and total > 0:
        rate = parse_errors / len(llm_results) if llm_results else 0
        print(
            f"WARNING: {parse_errors}/{len(llm_results)} LLM-evaluated "
            f"samples ({rate:.0%}) had parse errors and are excluded "
            f"from quality distributions.",
            file=sys.stderr,
        )

    # Normalized routing rates (% of total)
    no_mistake_rate = no_mistake / total if total else 0.0
    missed_rate = missed / total if total else 0.0
    hallucinated_rate = hallucinated / total if total else 0.0
    llm_evaluated_rate = len(llm_results) / total if total else 0.0
    parse_error_rate = parse_errors / len(llm_results) if llm_results else 0.0

    # Quality rates (% of judged)
    judged_count = len(mistake_cats)
    mistake_rates = {
        k: v / judged_count if judged_count else 0.0
        for k, v in mistake_dist.items()
    }
    feedback_rates = {
        k: v / judged_count if judged_count else 0.0
        for k, v in feedback_dist.items()
    }

    return JudgeMetrics(
        total=total,
        no_mistake=no_mistake,
        missed=missed,
        hallucinated=hallucinated,
        llm_evaluated=len(llm_results),
        judged=judged_count,
        parse_errors=parse_errors,
        mistake_distribution=mistake_dist,
        feedback_distribution=feedback_dist,
        no_mistake_rate=no_mistake_rate,
        missed_rate=missed_rate,
        hallucinated_rate=hallucinated_rate,
        llm_evaluated_rate=llm_evaluated_rate,
        parse_error_rate=parse_error_rate,
        mistake_rates=mistake_rates,
        feedback_rates=feedback_rates,
    )
