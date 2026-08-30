"""no_mistake/missed/hallucinated are the cases where neither side (or only
one side) has a mistake, so we skip the LLM call and record the outcome directly.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from music_evalkit.evaluation.llm_judge.metrics import compute_judge_metrics
from music_evalkit.evaluation.llm_judge.types import (
    JudgeMetrics,
    JudgeResult,
    JudgeSource,
)
from music_evalkit.evaluation.parsers import OpenEndedBasicPrediction
from music_evalkit.evaluation.utils import load_jsonl


def _no_mistake_result() -> JudgeResult:
    return JudgeResult(
        mistake=None,
        mistake_reasoning="Neither side has a mistake",
        feedback=None,
        feedback_reasoning="No feedback needed — no mistake",
        raw_response="",
        source=JudgeSource.NO_MISTAKE,
    )


def _missed_result() -> JudgeResult:
    return JudgeResult(
        mistake=None,
        mistake_reasoning="Model produced no mistake text to judge",
        feedback=None,
        feedback_reasoning="Model produced no feedback to judge",
        raw_response="",
        source=JudgeSource.MISSED,
    )


def _hallucinated_result() -> JudgeResult:
    return JudgeResult(
        mistake=None,
        mistake_reasoning="No GT mistake to compare against",
        feedback=None,
        feedback_reasoning="No GT feedback to compare against",
        raw_response="",
        source=JudgeSource.HALLUCINATED,
    )


def try_detect(
    pred: OpenEndedBasicPrediction, gt_mistake: str | None,
) -> JudgeResult | None:
    """Check if this sample can be resolved without the LLM judge.

    Returns a detection result if one or both sides lack mistake content,
    or None if both sides have content and the LLM judge should run.
    """
    if not pred.parse_success:
        return None  # can't tell what the model predicted, let LLM judge

    pred_has_content = pred.mistake is not None
    gt_has_content = gt_mistake is not None

    if not pred_has_content and not gt_has_content:
        return _no_mistake_result()
    if not pred_has_content and gt_has_content:
        return _missed_result()
    if pred_has_content and not gt_has_content:
        return _hallucinated_result()

    return None  # both sides have content → LLM judge needed


def load_resume(path: Path, resume: bool) -> list[dict]:
    """Load existing results for resume, cleaning corrupt trailing lines."""
    if not resume or not path.exists():
        return []

    existing = load_jsonl(path)
    if existing:
        missing = {"sample_id", "source"} - set(existing[0].keys())
        if missing:
            print(
                f"WARNING: Resume file has old schema (missing: {missing}). "
                f"Delete judge_results.jsonl and re-run without --resume.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Rewrite clean data (removes any corrupt trailing line)
    with open(path, "w") as f:
        for r in existing:
            f.write(json.dumps(r) + "\n")

    return existing


def extract_gt(gt) -> dict:
    """Extract and clean ground truth fields from a pandas row."""
    mistake = gt.get("mistake")
    feedback = gt.get("feedback")
    category = str(gt.get("mistake_category", ""))
    instruction = str(gt.get("instruction", ""))

    if pd.isna(mistake):
        mistake = None
    if pd.isna(feedback):
        feedback = None
    if pd.isna(category):
        category = ""
    if pd.isna(instruction):
        instruction = ""

    if mistake is not None and feedback is None:
        print(
            f"WARNING: GT has mistake but no feedback: "
            f"{str(mistake)[:80]}",
            file=sys.stderr,
        )

    return {
        "mistake": mistake,
        "feedback": feedback,
        "category": category,
        "instruction": instruction,
    }


def finalize(results: list[dict], output_dir: Path) -> JudgeMetrics:
    """Compute metrics and save to disk."""
    metrics = compute_judge_metrics(results)
    metrics_path = output_dir / "judge_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics.model_dump(), f, indent=2)
    return metrics
