"""None in y_pred means a parse error; excluded from metric computation and
tracked separately as parse_errors instead.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from music_evalkit.evaluation.semantic import SemanticSimilarity


class Metric(ABC):
    """Base class for evaluation metrics."""

    @abstractmethod
    def compute(self, y_pred: list, y_true: list) -> dict[str, Any]:
        """Compute the metric.

        Args:
            y_pred: Predicted values. None indicates a parse error.
            y_true: Ground-truth values (same length as y_pred).

        Returns:
            Dict mapping metric name to value.
        """



class MacroPRFMetric(Metric):
    """Macro-averaged precision, recall, and F1.

    Parse errors are excluded from the computation.
    """

    def compute(self, y_pred: list, y_true: list) -> dict[str, Any]:
        valid = [(p, t) for p, t in zip(y_pred, y_true) if p is not None]
        if not valid:
            return {"precision": None, "recall": None, "f1": None}
        preds, truths = zip(*valid)
        p, r, f, _ = precision_recall_fscore_support(
            truths, preds, average="macro", zero_division=0
        )
        return {"precision": float(p), "recall": float(r), "f1": float(f)}


class BinaryDetectionMetric(Metric):
    """Binary precision, recall, F1, and accuracy for mistake detection.

    y_pred and y_true should be bool (True = mistake present). None = unknown.
    """

    def compute(self, y_pred: list[bool | None], y_true: list[bool]) -> dict[str, Any]:
        valid = [(p, t) for p, t in zip(y_pred, y_true) if p is not None]
        if not valid:
            return {"precision": None, "recall": None, "f1": None, "accuracy": 0.0, "correct": 0}
        preds, truths = zip(*valid)
        p, r, f, _ = precision_recall_fscore_support(
            truths, preds, average="binary", zero_division=0
        )
        correct = sum(p == t for p, t in zip(preds, truths))
        return {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "accuracy": float(accuracy_score(truths, preds)),
            "correct": correct,
        }


class ConfusionMatrixMetric(Metric):
    """Confusion matrix returned as ``'actual|predicted': count`` dict.

    Parse errors (None predictions) are excluded.
    """

    def compute(self, y_pred: list, y_true: list) -> dict[str, Any]:
        cm: dict[str, int] = {}
        for pred, truth in zip(y_pred, y_true):
            if pred is None:
                continue
            key = f"{truth}|{pred}"
            cm[key] = cm.get(key, 0) + 1
        return {"confusion_matrix": cm}


class SemanticSimilarityMetric(Metric):
    """Mean SBERT cosine similarity with null-value handling.

    None values represent no_mistake samples:
    - both None → 1.0  (model correctly predicts no mistake)
    - one None  → 0.0  (mismatch)
    - both text → cosine similarity

    Args:
        semantic: Initialised SemanticSimilarity instance to reuse.
    """

    def __init__(self, semantic: SemanticSimilarity) -> None:
        self._semantic = semantic

    def compute(
        self, y_pred: list[str | None], y_true: list[str | None]
    ) -> dict[str, Any]:
        similarities: list[float] = []
        null_types: list[str] = []
        for pred, truth in zip(y_pred, y_true):
            sim, null_type = self._semantic.similarity_with_null_handling(pred, truth)
            similarities.append(sim)
            null_types.append(null_type)
        avg = float(np.mean(similarities)) if similarities else None
        return {
            "avg_similarity": avg,
            "similarities": similarities,
            "null_types": null_types,
        }
