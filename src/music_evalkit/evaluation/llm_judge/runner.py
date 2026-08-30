import json
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from music_evalkit.evaluation.llm_judge.helpers import (
    extract_gt,
    finalize,
    load_resume,
    try_detect,
)
from music_evalkit.evaluation.llm_judge.judge import PERMANENT_ERRORS, LLMJudge
from music_evalkit.evaluation.llm_judge.types import JudgeMetrics, JudgeResult, JudgeSource
from music_evalkit.evaluation.parsers import (
    parse_open_ended_recover,
    parse_open_ended_strict,
)
from music_evalkit.evaluation.utils import load_jsonl


MAX_CONSECUTIVE_FAILURES = 5


def run_judge(
    judge: LLMJudge,
    results_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
    resume: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    mode: str = "recover",
) -> JudgeMetrics:
    """Run LLM-as-a-Judge on open-ended inference results.

    Args:
        judge: Configured LLMJudge instance.
        results_path: Path to inference_results.jsonl.
        ground_truth_path: Path to ground truth samples.csv.
        output_dir: Directory for judge_results.jsonl and judge_metrics.json.
        resume: If True, skip already-judged samples.
        on_progress: Optional callback(done, total) for progress reporting.

    Returns:
        JudgeMetrics with detection breakdown and quality distributions.

    Raises:
        SystemExit: On fatal errors (auth, too many failures, GT mismatch).
    """
    results = load_jsonl(results_path)
    ground_truth = pd.read_csv(ground_truth_path, dtype={"id": str})
    gt_lookup = {str(row["id"]): row for _, row in ground_truth.iterrows()}

    output_dir.mkdir(parents=True, exist_ok=True)
    judge_results_path = output_dir / "judge_results.jsonl"

    # Resume
    existing = load_resume(judge_results_path, resume)
    judged_ids = {r["sample_id"] for r in existing}
    to_judge = [r for r in results if r["sample_id"] not in judged_ids]

    if not to_judge and existing:
        return finalize(existing, output_dir)

    # Main loop
    file_mode = "a" if existing else "w"
    all_results = list(existing)
    consecutive_failures = 0
    skipped_no_gt = 0

    with open(judge_results_path, file_mode) as f:
        for i, result in enumerate(to_judge):
            sample_id = result["sample_id"]

            gt = gt_lookup.get(str(sample_id))
            if gt is None:
                skipped_no_gt += 1
                continue

            gt_fields = extract_gt(gt)
            if mode == "strict":
                pred = parse_open_ended_strict(result["response_text"])
            else:
                pred = parse_open_ended_recover(result["response_text"])

            # Skip LLM if either side lacks mistake content
            judge_result = try_detect(pred, gt_fields["mistake"])
            if judge_result is None:
                try:
                    if pred.parse_success and pred.mistake is not None:
                        judge_result = judge.judge(
                            pred_reason=pred.reason,
                            pred_mistake=pred.mistake,
                            pred_feedback=pred.feedback or "",
                            ground_truth_mistake=gt_fields["mistake"] or "",
                            ground_truth_feedback=gt_fields["feedback"] or "",
                            instruction=gt_fields["instruction"],
                        )
                    else:
                        judge_result = judge.judge_raw(
                            raw_response=result["response_text"],
                            ground_truth_mistake=gt_fields["mistake"] or "",
                            ground_truth_feedback=gt_fields["feedback"] or "",
                            instruction=gt_fields["instruction"],
                        )
                    consecutive_failures = 0
                except PERMANENT_ERRORS as e:
                    print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
                    raise SystemExit(1)
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(
                            f"\nFATAL: {consecutive_failures} consecutive failures. "
                            f"Last: {type(e).__name__}: {e}",
                            file=sys.stderr,
                        )
                        raise SystemExit(1)
                    print(f"  Error judging {sample_id}: {type(e).__name__}: {e}")
                    judge_result = JudgeResult(
                        mistake=None,
                        mistake_reasoning=f"Judge error: {e}",
                        feedback=None,
                        feedback_reasoning=f"Judge error: {e}",
                        raw_response="",
                        source=JudgeSource.LLM,
                    )

            row = {"sample_id": sample_id, **judge_result.model_dump()}
            f.write(json.dumps(row) + "\n")
            f.flush()
            all_results.append(row)

            if on_progress and ((i + 1) % 10 == 0 or (i + 1) == len(to_judge)):
                on_progress(len(judged_ids) + i + 1, len(results))

    # Check ground truth skip rate
    if skipped_no_gt > 0:
        rate = skipped_no_gt / len(to_judge) if to_judge else 0
        print(f"WARNING: {skipped_no_gt} samples had no ground truth match", file=sys.stderr)
        if rate > 0.1:
            print("FATAL: >10% skipped. Check for ID format mismatch.", file=sys.stderr)
            raise SystemExit(1)

    return finalize(all_results, output_dir)
