"""BaseEvaluator handles the shared data loading/joining; MCQ/Pairwise/OpenEnded
subclass it. Result and metrics types are Pydantic models so cross-field
invariants get enforced by validators instead of scattered asserts.
"""

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, model_validator

from music_evalkit.evaluation.metrics import (
    BinaryDetectionMetric,
    ConfusionMatrixMetric,
    MacroPRFMetric,
    SemanticSimilarityMetric,
)
from music_evalkit.evaluation.parsers import (
    parse_mcq_response,
    parse_open_ended_recover,
    parse_open_ended_strict,
    parse_pairwise_response,
    text_to_letter,
)
from music_evalkit.evaluation.semantic import SemanticSimilarity


class MCQResult(BaseModel):
    """Evaluation result for a single MCQ sample."""

    sample_id: str
    category: str
    predicted_letter: str | None
    predicted_text: str | None
    ground_truth_letter: str | None
    ground_truth_text: str
    correct: bool
    parse_error: bool


class PairwiseResult(BaseModel):
    """Evaluation result for a single pairwise sample."""

    sample_id: str
    category: str
    predicted: str | None
    ground_truth: str
    correct: bool
    parse_error: bool


class OpenEndedBasicResult(BaseModel):
    """Evaluation result for a single open-ended sample."""

    sample_id: str
    ground_truth_category: str
    predicted_category: str | None
    mistake_similarity: float | None
    feedback_similarity: float | None
    null_match_type: Literal["both_null", "pred_null", "truth_null", "compared"]
    parse_error: bool


class CategoryAccuracy(BaseModel):
    """Per-category accuracy breakdown (MCQ / pairwise)."""

    correct: int
    total: int
    accuracy: float


class CategorySimilarity(BaseModel):
    """Per-category semantic similarity breakdown (open-ended)."""

    avg_similarity: float
    total: int


class EvaluationMetrics(BaseModel):
    """Aggregated evaluation metrics.

    Fields used by all tasks:
        total, correct, accuracy, parse_errors, precision, recall, f1

    MCQ / Pairwise:
        per_category_accuracy  — per-category correct/total/accuracy
        confusion_matrix, confusion_matrix_single, confusion_matrix_double

    Open-ended:
        per_category_similarity — per-category average SBERT similarity
        avg_mistake_similarity, avg_feedback_similarity
    """

    total: int
    correct: int = 0
    accuracy: float = 0.0
    parse_errors: int = 0

    # Classification metrics (macro for MCQ/pairwise; binary for open-ended)
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

    # MCQ / Pairwise
    per_category_accuracy: dict[str, CategoryAccuracy] = {}

    # Open-ended
    per_category_similarity: dict[str, CategorySimilarity] = {}
    avg_mistake_similarity: float | None = None
    avg_feedback_similarity: float | None = None

    # Confusion matrices: {"actual|predicted": count}
    confusion_matrix: dict[str, int] = {}
    confusion_matrix_single: dict[str, int] = {}
    confusion_matrix_double: dict[str, int] = {}

    @model_validator(mode="after")
    def check_consistency(self) -> "EvaluationMetrics":
        if self.total > 0:
            if self.correct > self.total:
                raise ValueError(f"correct ({self.correct}) > total ({self.total})")
            expected = self.correct / self.total
            if abs(self.accuracy - expected) > 1e-9:
                raise ValueError(
                    f"accuracy ({self.accuracy:.6f}) is inconsistent with "
                    f"correct/total ({self.correct}/{self.total} = {expected:.6f})"
                )
        prf = (self.precision, self.recall, self.f1)
        if not (all(v is None for v in prf) or all(v is not None for v in prf)):
            raise ValueError("precision, recall, and f1 must all be None or all be set")
        return self


class BaseEvaluator(ABC):
    """Shared data-loading utilities and abstract interface."""

    @abstractmethod
    def evaluate(
        self,
        results_path: Path,
        ground_truth_path: Path,
    ) -> tuple[list, EvaluationMetrics]:
        """Evaluate inference results against ground truth.

        Args:
            results_path: Path to inference_results.jsonl.
            ground_truth_path: Path to samples.csv.

        Returns:
            Tuple of (per-sample results list, aggregated EvaluationMetrics).
        """

    @staticmethod
    def _load_and_join(
        results_path: Path,
        ground_truth_path: Path,
        required_columns: list[str] | None = None,
    ) -> list[tuple[dict, dict]]:
        """Load inference JSONL + ground truth CSV and join on sample_id == id.

        The CSV is read with dtype={"id": str} to prevent int/string type
        mismatches that silently drop every sample.

        Args:
            results_path: Path to inference_results.jsonl.
            ground_truth_path: Path to samples.csv.
            required_columns: If provided, validate that these columns exist
                in the ground truth CSV.

        Returns:
            List of (result_dict, gt_row_dict) pairs.

        Raises:
            ValueError: If any inference result has no matching ground truth row,
                or if the result set is empty after loading.
        """
        results = BaseEvaluator._load_jsonl(results_path)
        if not results:
            raise ValueError(f"No inference results found in {results_path}")

        gt_df = pd.read_csv(ground_truth_path, dtype={"id": str})

        if required_columns:
            missing = [c for c in required_columns if c not in gt_df.columns]
            if missing:
                raise ValueError(
                    f"Ground truth CSV is missing required columns: {missing}. "
                    f"Available columns: {list(gt_df.columns)}"
                )

        gt_lookup: dict[str, dict] = gt_df.set_index("id").to_dict("index")

        matched: list[tuple[dict, dict]] = []
        unmatched: list[str] = []

        for result in results:
            sample_id = str(result["sample_id"])
            gt = gt_lookup.get(sample_id)
            if gt is None:
                unmatched.append(sample_id)
            else:
                matched.append((result, gt))

        if unmatched:
            raise ValueError(
                f"{len(unmatched)}/{len(results)} inference results had no matching "
                f"ground truth entry. First unmatched IDs: {unmatched[:5]}.\n"
                "Common cause: ID type mismatch or a pod sync without re-preprocessing. "
                "Check that the ground truth CSV matches the one used for inference."
            )

        return matched

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        """Load a JSONL file, warning and skipping malformed lines."""
        results = []
        skipped = 0
        total_lines = 0
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    skipped += 1
                    print(
                        f"WARNING: Skipping malformed JSON at {path}:{line_num}: {e}",
                        file=sys.stderr,
                    )
        if skipped > 0:
            skip_rate = skipped / total_lines
            print(
                f"WARNING: Skipped {skipped}/{total_lines} malformed lines "
                f"({skip_rate:.1%}) in {path}",
                file=sys.stderr,
            )
            if skip_rate > 0.05:
                raise ValueError(
                    f"Too many malformed lines ({skipped}/{total_lines}, "
                    f"{skip_rate:.1%}). File may be corrupted: {path}"
                )
        return results

    @staticmethod
    def _per_category_accuracy(
        categories: list[str], correct_flags: list[bool]
    ) -> dict[str, CategoryAccuracy]:
        """Aggregate per-category correct/total counts and accuracy."""
        acc: dict[str, dict] = {}
        for cat, correct in zip(categories, correct_flags):
            if cat not in acc:
                acc[cat] = {"correct": 0, "total": 0}
            acc[cat]["total"] += 1
            if correct:
                acc[cat]["correct"] += 1
        return {
            k: CategoryAccuracy(accuracy=v["correct"] / v["total"], **v)
            for k, v in acc.items()
        }

    @staticmethod
    def _warn_parse_error_rate(parse_errors: int, total: int, task_name: str) -> None:
        """Warn if parse error rate is dangerously high."""
        if total == 0:
            return
        rate = parse_errors / total
        if rate > 0.5:
            print(
                f"CRITICAL: {task_name} has {rate:.0%} parse error rate "
                f"({parse_errors}/{total}). Results are likely invalid.",
                file=sys.stderr,
            )
        elif rate > 0.1:
            print(
                f"WARNING: {task_name} has {rate:.0%} parse error rate "
                f"({parse_errors}/{total}). Metrics may be unreliable.",
                file=sys.stderr,
            )


class MCQEvaluator(BaseEvaluator):
    """Evaluator for the multiple-choice task.

    Metrics: accuracy, macro P/R/F1, confusion matrices (combined, single,
    double audio), per-category accuracy breakdown.
    """

    REQUIRED_COLUMNS = ["options", "answer", "mistake_category"]

    def __init__(self) -> None:
        self._prf = MacroPRFMetric()
        self._cm = ConfusionMatrixMetric()

    def evaluate(
        self,
        results_path: Path,
        ground_truth_path: Path,
    ) -> tuple[list[MCQResult], EvaluationMetrics]:
        matched = self._load_and_join(
            results_path, ground_truth_path, required_columns=self.REQUIRED_COLUMNS
        )

        sample_results: list[MCQResult] = []
        y_pred: list[str | None] = []
        y_true: list[str] = []
        single_pred: list[str | None] = []
        single_true: list[str] = []
        double_pred: list[str | None] = []
        double_true: list[str] = []
        categories: list[str] = []
        correct_flags: list[bool] = []

        for result, gt in matched:
            options = [opt.strip() for opt in gt["options"].split(",")]
            gt_text = gt["answer"].strip()
            gt_letter = text_to_letter(gt_text, options)

            pred = parse_mcq_response(result["response_text"], options)
            # None == None would be True — guard against missing gt_letter
            correct = (
                pred.letter is not None
                and gt_letter is not None
                and pred.letter == gt_letter
            )
            parse_error = pred.letter is None
            pred_text = pred.option_text.strip() if pred.option_text is not None else None

            category = gt["mistake_category"]

            y_pred.append(pred_text)
            y_true.append(gt_text)
            categories.append(category)
            correct_flags.append(correct)

            audio_setting = str(gt.get("audio_setting", "")).strip().lower()
            if audio_setting == "single":
                single_pred.append(pred_text)
                single_true.append(gt_text)
            elif audio_setting == "double":
                double_pred.append(pred_text)
                double_true.append(gt_text)

            sample_results.append(
                MCQResult(
                    sample_id=result["sample_id"],
                    category=category,
                    predicted_letter=pred.letter,
                    predicted_text=pred.option_text,
                    ground_truth_letter=gt_letter,
                    ground_truth_text=gt_text,
                    correct=correct,
                    parse_error=parse_error,
                )
            )

        prf = self._prf.compute(y_pred, y_true)
        cm = self._cm.compute(y_pred, y_true)
        cm_single = self._cm.compute(single_pred, single_true)
        cm_double = self._cm.compute(double_pred, double_true)

        parse_errors = sum(1 for r in sample_results if r.parse_error)
        correct_count = sum(1 for r in sample_results if r.correct)
        total = len(sample_results)

        self._warn_parse_error_rate(parse_errors, total, "mcq")

        return sample_results, EvaluationMetrics(
            total=total,
            correct=correct_count,
            accuracy=correct_count / total if total > 0 else 0.0,
            parse_errors=parse_errors,
            precision=prf["precision"],
            recall=prf["recall"],
            f1=prf["f1"],
            per_category_accuracy=self._per_category_accuracy(categories, correct_flags),
            confusion_matrix=cm["confusion_matrix"],
            confusion_matrix_single=cm_single["confusion_matrix"],
            confusion_matrix_double=cm_double["confusion_matrix"],
        )


class PairwiseEvaluator(BaseEvaluator):
    """Evaluator for the pairwise comparison task.

    Metrics: accuracy, macro P/R/F1, confusion matrix, per-category accuracy.
    """

    REQUIRED_COLUMNS = ["label", "category"]

    def __init__(self) -> None:
        self._prf = MacroPRFMetric()
        self._cm = ConfusionMatrixMetric()

    def evaluate(
        self,
        results_path: Path,
        ground_truth_path: Path,
    ) -> tuple[list[PairwiseResult], EvaluationMetrics]:
        matched = self._load_and_join(
            results_path, ground_truth_path, required_columns=self.REQUIRED_COLUMNS
        )

        sample_results: list[PairwiseResult] = []
        y_pred: list[str | None] = []
        y_true: list[str] = []
        categories: list[str] = []
        correct_flags: list[bool] = []

        for result, gt in matched:
            _raw = str(gt["label"]).strip().upper()
            gt_label = {"1": "A", "2": "B"}.get(_raw, _raw)
            category = gt["category"]

            pred = parse_pairwise_response(result["response_text"])
            correct = pred.choice == gt_label
            parse_error = pred.choice is None

            y_pred.append(pred.choice)
            y_true.append(gt_label)
            categories.append(category)
            correct_flags.append(correct)

            sample_results.append(
                PairwiseResult(
                    sample_id=result["sample_id"],
                    category=category,
                    predicted=pred.choice,
                    ground_truth=gt_label,
                    correct=correct,
                    parse_error=parse_error,
                )
            )

        prf = self._prf.compute(y_pred, y_true)
        cm = self._cm.compute(y_pred, y_true)

        parse_errors = sum(1 for r in sample_results if r.parse_error)
        correct_count = sum(1 for r in sample_results if r.correct)
        total = len(sample_results)

        self._warn_parse_error_rate(parse_errors, total, "pairwise")

        return sample_results, EvaluationMetrics(
            total=total,
            correct=correct_count,
            accuracy=correct_count / total if total > 0 else 0.0,
            parse_errors=parse_errors,
            precision=prf["precision"],
            recall=prf["recall"],
            f1=prf["f1"],
            per_category_accuracy=self._per_category_accuracy(categories, correct_flags),
            confusion_matrix=cm["confusion_matrix"],
        )


class OpenEndedEvaluator(BaseEvaluator):
    """Evaluator for the open-ended task.

    Metrics:
    - Binary mistake detection: accuracy (all samples), P/R/F1 (valid predictions only)
    - Category classification: accuracy, confusion matrix, per-category accuracy
    - SBERT semantic similarity for mistake and feedback (per-pair encoding)
    - Per-category average similarity breakdown
    """

    REQUIRED_COLUMNS = ["mistake_category"]

    def __init__(self, sbert_model: str = "all-MiniLM-L6-v2", mode: str = "recover") -> None:
        if mode not in ("strict", "recover"):
            raise ValueError(f"Invalid mode: {mode!r}. Must be 'strict' or 'recover'.")
        self._mode = mode
        semantic = SemanticSimilarity(model_name=sbert_model)
        self._similarity = SemanticSimilarityMetric(semantic)
        self._detection = BinaryDetectionMetric()
        self._prf = MacroPRFMetric()
        self._cm = ConfusionMatrixMetric()

    def evaluate(
        self,
        results_path: Path,
        ground_truth_path: Path,
    ) -> tuple[list[OpenEndedBasicResult], EvaluationMetrics]:
        matched = self._load_and_join(
            results_path, ground_truth_path, required_columns=self.REQUIRED_COLUMNS
        )

        # Collect all data first so metrics can be computed in a single pass.
        sample_ids: list[str] = []
        gt_categories: list[str] = []
        pred_categories: list[str | None] = []
        detection_pred: list[bool | None] = []
        detection_true: list[bool] = []
        mistake_pred: list[str | None] = []
        mistake_true: list[str | None] = []
        feedback_pred: list[str | None] = []
        feedback_true: list[str | None] = []
        parse_error_flags: list[bool] = []

        for result, gt in matched:
            gt_category = (
                str(gt["mistake_category"]).strip().lower()
                if not pd.isna(gt.get("mistake_category"))
                else ""
            )
            gt_mistake = None if pd.isna(gt.get("mistake")) else gt.get("mistake")
            gt_feedback = None if pd.isna(gt.get("feedback")) else gt.get("feedback")

            if self._mode == "strict":
                pred = parse_open_ended_strict(result["response_text"])
            else:
                pred = parse_open_ended_recover(result["response_text"])

            # Normalize predicted category
            pred_cat = pred.category.strip().lower() if pred.category else None

            sample_ids.append(str(result["sample_id"]))
            gt_categories.append(gt_category)
            pred_categories.append(pred_cat)
            # Derive detection from whether the model described a mistake.
            # Partial regex parses (parse_success=False) are treated as unknown.
            if not pred.parse_success:
                detection_pred.append(None)
            else:
                detection_pred.append(pred.mistake is not None)
            detection_true.append(gt_category != "no_mistake")
            mistake_pred.append(pred.mistake)
            mistake_true.append(gt_mistake)
            feedback_pred.append(pred.feedback)
            feedback_true.append(gt_feedback)
            parse_error_flags.append(not pred.parse_success)

        # Compute metrics.
        det = self._detection.compute(detection_pred, detection_true)
        mistake_sim = self._similarity.compute(mistake_pred, mistake_true)
        feedback_sim = self._similarity.compute(feedback_pred, feedback_true)

        # Category classification metrics.
        cat_prf = self._prf.compute(pred_categories, gt_categories)
        cat_cm = self._cm.compute(pred_categories, gt_categories)
        cat_correct_flags = [
            p is not None and p == t
            for p, t in zip(pred_categories, gt_categories)
        ]

        # Build per-sample results from batch outputs.
        sample_results = [
            OpenEndedBasicResult(
                sample_id=sample_ids[i],
                ground_truth_category=gt_categories[i],
                predicted_category=pred_categories[i],
                mistake_similarity=mistake_sim["similarities"][i],
                feedback_similarity=feedback_sim["similarities"][i],
                null_match_type=mistake_sim["null_types"][i],
                parse_error=parse_error_flags[i],
            )
            for i in range(len(sample_ids))
        ]

        # Accuracy over ALL samples (None predictions count as wrong), consistent
        # with MCQ/pairwise.  PRF is over valid (non-None) predictions only.
        correct_count = sum(
            1 for p, t in zip(detection_pred, detection_true)
            if p is not None and p == t
        )
        total = len(sample_results)

        self._warn_parse_error_rate(sum(parse_error_flags), total, "open_ended")

        # Per-category average mistake similarity.
        per_cat: dict[str, list[float]] = {}
        for cat, sim in zip(gt_categories, mistake_sim["similarities"]):
            per_cat.setdefault(cat, []).append(sim)
        per_category_similarity = {
            cat: CategorySimilarity(
                avg_similarity=sum(sims) / len(sims),
                total=len(sims),
            )
            for cat, sims in per_cat.items()
        }

        return sample_results, EvaluationMetrics(
            total=total,
            correct=correct_count,
            accuracy=correct_count / total if total > 0 else 0.0,
            parse_errors=sum(parse_error_flags),
            precision=det["precision"],
            recall=det["recall"],
            f1=det["f1"],
            avg_mistake_similarity=mistake_sim["avg_similarity"],
            avg_feedback_similarity=feedback_sim["avg_similarity"],
            per_category_similarity=per_category_similarity,
            per_category_accuracy=self._per_category_accuracy(
                gt_categories, cat_correct_flags
            ),
            confusion_matrix=cat_cm["confusion_matrix"],
        )
