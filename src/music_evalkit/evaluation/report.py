import json
from datetime import datetime
from pathlib import Path

from music_evalkit.evaluation.evaluator import EvaluationMetrics
from music_evalkit.evaluation.llm_judge import JudgeMetrics
from music_evalkit.evaluation.utils import cm_dict_to_dataframe


def generate_report(
    task_name: str,
    metrics: EvaluationMetrics,
    output_dir: Path,
    results_dir: Path,
    judge_metrics: JudgeMetrics | None = None,
) -> Path:
    """Generate a markdown evaluation report.

    Returns:
        Path to the written report file.
    """
    meta = _load_metadata(results_dir)
    lines: list[str] = []

    _add_header(lines, task_name, meta)
    _add_summary(lines, task_name, metrics, judge_metrics)
    _add_audio_setting_analysis(lines, task_name, metrics)
    _add_per_category(lines, task_name, metrics)

    if task_name == "open_ended" and judge_metrics:
        _add_judge_section(lines, judge_metrics)

    if metrics.confusion_matrix:
        _add_per_class_prf(lines, metrics)
        _add_prediction_distribution(lines, metrics)

    _add_confusion_matrices(lines, task_name, metrics)

    lines += [
        "---",
        "",
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def _load_metadata(results_dir: Path) -> dict:
    parts = results_dir.resolve().parts
    meta = {
        "run_id": parts[-1] if len(parts) >= 1 else "unknown",
        "provider": parts[-2] if len(parts) >= 2 else "unknown",
        "task": parts[-3] if len(parts) >= 3 else "unknown",
        "model": None,
        "prompt_hash": None,
    }

    metadata_path = results_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                run_meta = json.load(f)
            meta["run_id"] = run_meta.get("run_id", meta["run_id"])
            meta["provider"] = run_meta.get("provider", meta["provider"])
            meta["model"] = run_meta.get("model")
            meta["prompt_hash"] = run_meta.get("prompt_hash")
            meta["task"] = run_meta.get("task", meta["task"])
        except (json.JSONDecodeError, KeyError):
            pass

    return meta


def _add_header(lines: list[str], task_name: str, meta: dict) -> None:
    display_task = task_name.replace("_", " ").title()
    lines += [
        f"# {display_task} — {meta.get('model') or meta['provider']}",
        "",
        "| | |",
        "|---|---|",
        f"| **Run ID** | `{meta['run_id']}` |",
        f"| **Provider** | `{meta['provider']}` |",
    ]
    if meta["model"]:
        lines.append(f"| **Model** | `{meta['model']}` |")
    if meta["prompt_hash"]:
        lines.append(f"| **Prompt hash** | `{meta['prompt_hash']}` |")
    lines += ["", ""]


def _add_summary(
    lines: list[str],
    task_name: str,
    metrics: EvaluationMetrics,
    judge_metrics: JudgeMetrics | None = None,
) -> None:
    lines += ["## Summary", ""]

    if task_name in ("mcq", "pairwise"):
        lines += [
            f"Overall performance on the {task_name.upper()} task. "
            f"Accuracy measures exact-match between predicted and ground truth answers. "
            f"Precision, recall, and F1 are macro-averaged across all classes.",
            "",
        ]
    else:
        lines += [
            "Overall performance on the open-ended task. Detection accuracy measures "
            "whether the model correctly identified the presence/absence of a mistake. "
            "SBERT similarity measures semantic overlap between predicted and ground truth "
            "mistake descriptions and feedback (1.0 = identical meaning, 0.0 = unrelated).",
            "",
        ]

    parse_pct = f" ({metrics.parse_errors / metrics.total:.1%})" if metrics.total else ""

    if task_name in ("mcq", "pairwise"):
        rows = [
            ("Total samples", str(metrics.total)),
            ("Parse errors", f"{metrics.parse_errors}{parse_pct}"),
            ("Correct", str(metrics.correct)),
            ("**Accuracy**", f"**{metrics.accuracy:.1%}**"),
        ]
        if metrics.precision is not None:
            rows.append(("Precision (macro)", f"{metrics.precision:.3f}"))
        if metrics.recall is not None:
            rows.append(("Recall (macro)", f"{metrics.recall:.3f}"))
        if metrics.f1 is not None:
            rows.append(("**F1 (macro)**", f"**{metrics.f1:.3f}**"))

    else:  # open_ended
        rows = [
            ("Total samples", str(metrics.total)),
            ("Parse errors", f"{metrics.parse_errors}{parse_pct}"),
            ("Correct detections", str(metrics.correct)),
            ("**Detection accuracy**", f"**{metrics.accuracy:.1%}**"),
        ]
        if metrics.precision is not None:
            rows.append(("Precision", f"{metrics.precision:.3f}"))
        if metrics.recall is not None:
            rows.append(("Recall", f"{metrics.recall:.3f}"))
        if metrics.f1 is not None:
            rows.append(("**F1**", f"**{metrics.f1:.3f}**"))
        if metrics.avg_mistake_similarity is not None:
            rows.append(("SBERT mistake sim.", f"{metrics.avg_mistake_similarity:.3f}"))
        if metrics.avg_feedback_similarity is not None:
            rows.append(("SBERT feedback sim.", f"{metrics.avg_feedback_similarity:.3f}"))

        # Inline key judge stats if available
        if judge_metrics and judge_metrics.judged > 0:
            jm = judge_metrics
            correct_r = jm.mistake_rates.get("correct", 0.0)
            helpful_r = jm.feedback_rates.get("helpful", 0.0)
            rows.append(("Judge: Correct mistake", f"{correct_r:.1%} ({jm.mistake_distribution.get('correct', 0)}/{jm.judged})"))
            rows.append(("Judge: Helpful feedback", f"{helpful_r:.1%} ({jm.feedback_distribution.get('helpful', 0)}/{jm.judged})"))

    lines += ["| Metric | Value |", "|--------|------:|"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    lines += ["", ""]


def _add_audio_setting_analysis(
    lines: list[str], task_name: str, metrics: EvaluationMetrics
) -> None:
    if task_name in ("mcq", "pairwise"):
        acc_single = _accuracy_from_cm(metrics.confusion_matrix_single)
        acc_double = _accuracy_from_cm(metrics.confusion_matrix_double)
        n_single = sum(metrics.confusion_matrix_single.values()) if metrics.confusion_matrix_single else 0
        n_double = sum(metrics.confusion_matrix_double.values()) if metrics.confusion_matrix_double else 0
        if acc_single is None and acc_double is None:
            return

        lines += [
            "## Results by Audio Setting",
            "",
            "Single-audio samples provide only the student recording. "
            "Double-audio samples include both a reference performance and the student recording, "
            "separated by a beep.",
            "",
            "| Setting | Samples | Accuracy |",
            "|---------|--------:|---------:|",
        ]
        if acc_single is not None:
            lines.append(f"| Single audio | {n_single} | {acc_single:.1%} |")
        if acc_double is not None:
            lines.append(f"| Double audio | {n_double} | {acc_double:.1%} |")
        lines += ["", ""]


def _add_per_category(
    lines: list[str], task_name: str, metrics: EvaluationMetrics
) -> None:
    if task_name in ("mcq", "pairwise") and metrics.per_category_accuracy:
        lines += [
            "## Per-Category Accuracy",
            "",
            "Accuracy broken down by mistake category (e.g., pitch, rhythm, harmony). "
            "Sorted from highest to lowest accuracy.",
            "",
            "| Category | Correct | Total | Accuracy |",
            "|----------|--------:|------:|---------:|",
        ]
        for cat, data in sorted(
            metrics.per_category_accuracy.items(), key=lambda x: -x[1].accuracy
        ):
            lines.append(
                f"| {cat} | {data.correct} | {data.total} | {data.accuracy:.1%} |"
            )
        lines += ["", ""]

    elif task_name == "open_ended":
        if metrics.per_category_accuracy:
            lines += [
                "## Per-Category Classification",
                "",
                "How often the model predicted the correct mistake category "
                "(e.g., pitch, rhythm, harmony). Sorted from highest to lowest accuracy.",
                "",
                "| Category | Correct | Total | Accuracy |",
                "|----------|--------:|------:|---------:|",
            ]
            for cat, data in sorted(
                metrics.per_category_accuracy.items(), key=lambda x: -x[1].accuracy
            ):
                lines.append(
                    f"| {cat} | {data.correct} | {data.total} | {data.accuracy:.1%} |"
                )
            lines += ["", ""]

        if metrics.per_category_similarity:
            lines += [
                "## Per-Category Similarity (SBERT)",
                "",
                "Average semantic similarity between predicted and ground truth mistake descriptions, "
                "broken down by category. Higher values indicate closer semantic match.",
                "",
                "| Category | Samples | Mistake Sim. |",
                "|----------|--------:|-------------:|",
            ]
            for cat, data in sorted(
                metrics.per_category_similarity.items(),
                key=lambda x: -x[1].avg_similarity,
            ):
                lines.append(
                    f"| {cat} | {data.total} | {data.avg_similarity:.3f} |"
                )
            lines += ["", ""]


def _add_judge_section(lines: list[str], jm: JudgeMetrics) -> None:
    lines += [
        "## LLM-as-a-Judge",
        "",
        "A secondary LLM evaluates the quality of model predictions by comparing them "
        "against ground truth on two dimensions: mistake description (how well does the predicted "
        "mistake match?) and feedback quality (is the corrective advice useful?). Only samples where "
        "both the prediction and ground truth contain mistake content are sent to the judge.",
        "",
    ]

    # Sample routing
    total = jm.total
    lines += [
        "### Sample Routing",
        "",
        "Before reaching the judge LLM, samples are routed based on whether "
        "the prediction and ground truth contain mistake content.",
        "",
        "- **No mistake**: neither side contains a mistake — nothing to judge.",
        "- **False negative**: the ground truth has a mistake but the model produced none.",
        "- **False positive**: the model reported a mistake but the ground truth has none.",
        "- **LLM evaluated**: both sides have content — sent to the judge.",
        "",
        "| Outcome | Count | Rate |",
        "|---------|------:|-----:|",
    ]
    for label, count, rate in [
        ("No mistake", jm.no_mistake, jm.no_mistake_rate),
        ("False negative", jm.missed, jm.missed_rate),
        ("False positive", jm.hallucinated, jm.hallucinated_rate),
        ("LLM evaluated", jm.llm_evaluated, jm.llm_evaluated_rate),
    ]:
        lines.append(f"| {label} | {count} | {rate:.1%} |")
    lines += [
        f"| **Total** | **{total}** | |",
        "",
    ]

    if jm.parse_errors > 0:
        lines.append(
            f"*{jm.parse_errors} of {jm.llm_evaluated} LLM-evaluated samples "
            f"had parse errors and are excluded from quality ratings.*"
        )
        lines += ["", ""]

    if jm.judged == 0:
        return

    j = jm.judged

    # Mistake Description
    lines += [
        "### Mistake Description",
        "",
        "How well does the predicted mistake match the ground truth?",
        "",
        "| Rating | Count | Rate |",
        "|--------|------:|-----:|",
    ]
    for cat in ["correct", "partially_correct", "incorrect"]:
        c = jm.mistake_distribution.get(cat, 0)
        r = jm.mistake_rates.get(cat, 0.0)
        lines.append(f"| {cat} | {c} | {r:.1%} |")
    lines += ["", ""]

    # Feedback Quality
    lines += [
        "### Feedback Quality",
        "",
        "Is the corrective feedback specific and actionable?",
        "",
        "| Rating | Count | Rate |",
        "|--------|------:|-----:|",
    ]
    for cat in ["helpful", "generic", "unhelpful"]:
        c = jm.feedback_distribution.get(cat, 0)
        r = jm.feedback_rates.get(cat, 0.0)
        lines.append(f"| {cat} | {c} | {r:.1%} |")
    lines += ["", ""]


def _add_per_class_prf(lines: list[str], metrics: EvaluationMetrics) -> None:
    prf = _precision_recall_f1(metrics.confusion_matrix)
    if not prf:
        return

    support: dict[str, int] = {}
    for key, count in metrics.confusion_matrix.items():
        actual = key.split("|", 1)[0]
        support[actual] = support.get(actual, 0) + count

    lines += [
        "## Per-Class Precision / Recall / F1",
        "",
        "Per-class metrics computed from the confusion matrix. "
        "Precision = how often a predicted class is correct. "
        "Recall = how often an actual class is detected. "
        "Support = number of ground truth samples per class.",
        "",
        "| Category | P | R | F1 | Support |",
        "|----------|--:|--:|---:|--------:|",
    ]

    for cat, vals in sorted(prf.items(), key=lambda x: -(x[1].get("f1") or 0)):
        p = f"{vals['precision']:.2f}" if vals["precision"] is not None else "—"
        r = f"{vals['recall']:.2f}" if vals["recall"] is not None else "—"
        f1 = f"{vals['f1']:.2f}" if vals["f1"] is not None else "—"
        s = support.get(cat, 0)
        lines.append(f"| {cat} | {p} | {r} | {f1} | {s} |")

    lines += ["", ""]


def _add_prediction_distribution(lines: list[str], metrics: EvaluationMetrics) -> None:
    pred_dist = _prediction_distribution(metrics.confusion_matrix)
    if not pred_dist:
        return

    total_parsed = sum(pred_dist.values())

    gt_dist: dict[str, int] = {}
    for key, count in metrics.confusion_matrix.items():
        actual = key.split("|", 1)[0]
        gt_dist[actual] = gt_dist.get(actual, 0) + count

    all_labels = sorted(set(pred_dist.keys()) | set(gt_dist.keys()))
    total_gt = sum(gt_dist.values())

    lines += [
        "## Prediction vs Ground Truth Distribution",
        "",
        "How often each category was predicted vs how often it actually appears. "
        "Large discrepancies indicate systematic bias (e.g., over-predicting a category).",
        "",
        "| Category | Predicted | (%) | Actual | (%) |",
        "|----------|----------:|----:|-------:|----:|",
    ]
    for label in all_labels:
        p_count = pred_dist.get(label, 0)
        g_count = gt_dist.get(label, 0)
        p_pct = f"{p_count / total_parsed:.1%}" if total_parsed else "—"
        g_pct = f"{g_count / total_gt:.1%}" if total_gt else "—"
        lines.append(f"| {label} | {p_count} | {p_pct} | {g_count} | {g_pct} |")

    lines += ["", ""]


def _add_confusion_matrices(
    lines: list[str], task_name: str, metrics: EvaluationMetrics
) -> None:
    matrices = [
        (metrics.confusion_matrix, "Overall"),
        (metrics.confusion_matrix_single, "Single Audio"),
        (metrics.confusion_matrix_double, "Double Audio"),
    ]

    any_rendered = False
    for cm_dict, title in matrices:
        cm_md = _confusion_matrix_md(cm_dict, title)
        if cm_md:
            if not any_rendered:
                lines += ["## Confusion Matrices", ""]
                any_rendered = True
            lines += cm_md + [""]


def _accuracy_from_cm(cm_dict: dict) -> float | None:
    if not cm_dict:
        return None
    correct = sum(v for k, v in cm_dict.items() if k.split("|")[0] == k.split("|")[1])
    total = sum(cm_dict.values())
    return correct / total if total > 0 else None


def _prediction_distribution(cm_dict: dict) -> dict[str, int]:
    dist: dict[str, int] = {}
    for key, count in cm_dict.items():
        predicted = key.split("|", 1)[1]
        dist[predicted] = dist.get(predicted, 0) + count
    return dist


def _precision_recall_f1(cm_dict: dict) -> dict[str, dict]:
    df = cm_dict_to_dataframe(cm_dict)
    if df.empty:
        return {}

    all_labels = list(df.index)
    result = {}
    for label in all_labels:
        tp = df.loc[label, label]
        fp = df[label].sum() - tp
        fn = df.loc[label].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        result[label] = {"precision": precision, "recall": recall, "f1": f1}

    return result


def _confusion_matrix_md(cm_dict: dict, title: str) -> list[str]:
    if not cm_dict:
        return []

    df = cm_dict_to_dataframe(cm_dict)
    if df.empty:
        return []

    all_labels = list(df.index)
    lines = [f"### {title}", ""]
    header = "| Actual \\ Predicted | " + " | ".join(all_labels) + " |"
    separator = "|---|" + "|".join(["---:"] * len(all_labels)) + "|"
    lines += [header, separator]
    for label in all_labels:
        row = " | ".join(str(df.loc[label, col]) for col in all_labels)
        lines.append(f"| **{label}** | {row} |")

    return lines
