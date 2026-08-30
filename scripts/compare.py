#!/usr/bin/env python3
"""CMI-Bench-style comparison tables across all models/tasks.

Usage:
    python scripts/compare.py
    python scripts/compare.py --results-dir data/results --output comparison.md
"""

import json
from pathlib import Path

import click
from dotenv import load_dotenv

from music_evalkit.evaluation.visualization import plot_judge_distributions, plot_confusion_matrix_grid

load_dotenv()


TASKS = ["mcq", "pairwise", "open_ended"]


@click.command()
@click.option(
    "--results-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/results"),
    help="Root results directory",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output markdown path (default: results-dir/comparison.md)",
)
def main(results_dir: Path, output: Path | None):
    """Generate cross-model comparison report."""
    if output is None:
        output = results_dir / "comparison.md"

    # Discover all runs
    runs = _discover_runs(results_dir)
    if not runs:
        click.echo("No evaluation results found.", err=True)
        raise SystemExit(1)

    models = sorted({r["model"] for r in runs.values()})
    click.echo(f"Found {len(runs)} runs across {len(models)} models")

    lines: list[str] = []
    _add_header(lines, models)
    _add_main_table(lines, runs, models)
    _add_audio_setting_table(lines, runs, models)
    _add_judge_table(lines, runs, models)
    _add_per_category_tables(lines, runs, models)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    click.echo(f"Report: {output}")

    # Judge distribution plot
    judge_data = {}
    for (task, provider), run in runs.items():
        if task == "open_ended" and run.get("judge_metrics"):
            judge_data[run["model"]] = run["judge_metrics"]
    if judge_data:
        plot_path = output.parent / "judge_distributions.png"
        plot_judge_distributions(judge_data, plot_path)
        click.echo(f"Judge plot: {plot_path}")

    # Combined confusion matrix grids
    _MODEL_ORDER = [
        "gemini-2.0-flash", "nvidia/audio-flamingo-3-hf",
        "qwen2.5-omni-7b", "nvidia/music-flamingo-hf",
    ]
    _MODEL_LABELS = {
        "gemini-2.0-flash": "Gemini 2.0 Flash",
        "nvidia/audio-flamingo-3-hf": "Audio Flamingo 3",
        "qwen2.5-omni-7b": "Qwen2.5-Omni",
        "nvidia/music-flamingo-hf": "Music Flamingo",
    }

    # MCQ grid — use standard run metrics
    mcq_entries = []
    for model in _MODEL_ORDER:
        run = _find_run(runs, "mcq", model)
        if run is None:
            continue
        cm_dict = run["metrics"].get("confusion_matrix")
        if cm_dict:
            mcq_entries.append((_MODEL_LABELS.get(model, model), cm_dict))
    if mcq_entries:
        grid_path = output.parent / "mcq_confusion_matrices_combined.png"
        plot_confusion_matrix_grid(mcq_entries, "Mcq Confusion Matrix (Normalised)", grid_path, normalize=True)
        click.echo(f"Grid plot: {grid_path}")

    # Open-ended grid — _discover_runs() already prefers recover/ metrics
    oe_entries = []
    for model in _MODEL_ORDER:
        run = _find_run(runs, "open_ended", model)
        if run is None:
            continue
        cm_dict = run["metrics"].get("confusion_matrix")
        if cm_dict:
            oe_entries.append((_MODEL_LABELS.get(model, model), cm_dict))
    if oe_entries:
        grid_path = output.parent / "open_ended_confusion_matrices_combined.png"
        plot_confusion_matrix_grid(oe_entries, "Open-ended Confusion Matrices (Normalised, After Recovery)", grid_path, normalize=True)
        click.echo(f"Grid plot: {grid_path}")


def _discover_runs(results_dir: Path) -> dict[tuple[str, str], dict]:
    """Discover all runs, keyed by (task, provider).

    For each task/provider, picks the latest run (by run_id timestamp).
    Returns dict with loaded metrics.
    """
    runs: dict[tuple[str, str], dict] = {}

    for task in TASKS:
        task_dir = results_dir / task
        if not task_dir.exists():
            continue

        for provider_dir in sorted(task_dir.iterdir()):
            if not provider_dir.is_dir():
                continue
            provider = provider_dir.name

            # Pick latest run
            run_dirs = sorted(
                [d for d in provider_dir.iterdir() if d.is_dir()],
                key=lambda d: d.name,
                reverse=True,
            )
            for run_dir in run_dirs:
                # open_ended evaluation output is nested under recover/ or
                # strict/ (see evaluate.py's --mode); prefer that over any
                # stale top-level file from before the nesting was introduced.
                metrics_dir = run_dir / "recover" if task == "open_ended" else run_dir
                if not (metrics_dir / "evaluation_metrics.json").exists():
                    metrics_dir = run_dir
                metrics_path = metrics_dir / "evaluation_metrics.json"
                if not metrics_path.exists():
                    continue

                with open(metrics_path) as f:
                    metrics = json.load(f)

                # Load judge metrics if available
                judge_metrics = None
                judge_path = metrics_dir / "judge_metrics.json"
                if judge_path.exists():
                    with open(judge_path) as f:
                        judge_metrics = json.load(f)

                # Load model name from metadata
                model = provider
                meta_path = run_dir / "run_metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    model = meta.get("model", provider)

                runs[(task, provider)] = {
                    "task": task,
                    "provider": provider,
                    "model": model,
                    "run_dir": run_dir,
                    "metrics": metrics,
                    "judge_metrics": judge_metrics,
                }
                break  # use latest run only

    return runs


def _add_header(lines: list[str], models: list[str]) -> None:
    lines += [
        "# Model Comparison",
        "",
        f"Comparing **{len(models)} models** across MCQ, pairwise, and open-ended tasks. "
        "Best result per metric is shown in **bold**.",
        "",
    ]


def _add_main_table(
    lines: list[str], runs: dict, models: list[str]
) -> None:
    """Main results table — tasks as row groups, models as columns."""
    lines += [
        "## Main Results",
        "",
    ]

    # Build model display names (short)
    model_names = _model_display_names(runs, models)
    header = "| Task | Metric | " + " | ".join(model_names) + " |"
    sep = "|------|--------|" + "|".join(["---:" for _ in models]) + "|"
    lines += [header, sep]

    # MCQ
    _add_task_rows(lines, runs, models, "mcq", [
        ("Accuracy ↑", lambda m: m.get("accuracy"), _fmt_3f),
        ("F1 (macro) ↑", lambda m: m.get("f1"), _fmt_3f),
        ("Parse errors", lambda m: m.get("parse_errors"), _fmt_int),
    ])

    # Pairwise
    _add_task_rows(lines, runs, models, "pairwise", [
        ("Accuracy ↑", lambda m: m.get("accuracy"), _fmt_3f),
        ("F1 (macro) ↑", lambda m: m.get("f1"), _fmt_3f),
        ("Parse errors", lambda m: m.get("parse_errors"), _fmt_int),
    ])

    # Open-ended
    _add_task_rows(lines, runs, models, "open_ended", [
        ("Det. accuracy ↑", lambda m: m.get("accuracy"), _fmt_3f),
        ("F1 ↑", lambda m: m.get("f1"), _fmt_3f),
        ("SBERT mistake ↑", lambda m: m.get("avg_mistake_similarity"), _fmt_3f),
        ("SBERT feedback ↑", lambda m: m.get("avg_feedback_similarity"), _fmt_3f),
        ("Parse errors", lambda m: m.get("parse_errors"), _fmt_int),
    ])

    lines += ["", ""]


def _add_task_rows(
    lines: list[str],
    runs: dict,
    models: list[str],
    task: str,
    metric_defs: list[tuple[str, callable, callable]],
) -> None:
    """Add rows for a single task to the main table."""
    display_task = task.replace("_", " ").title()

    for i, (metric_name, extractor, formatter) in enumerate(metric_defs):
        task_label = f"**{display_task}**" if i == 0 else ""

        values = []
        for model in models:
            run = _find_run(runs, task, model)
            if run is None:
                values.append(None)
            else:
                values.append(extractor(run["metrics"]))

        # Bold the best value
        is_higher_better = "↑" in metric_name
        is_lower_better = "↓" in metric_name
        best_idx = _find_best(values, higher_better=is_higher_better) if (is_higher_better or is_lower_better) else None

        cells = []
        for j, v in enumerate(values):
            formatted = formatter(v) if v is not None else "—"
            if j == best_idx:
                formatted = f"**{formatted}**"
            cells.append(formatted)

        lines.append(f"| {task_label} | {metric_name} | " + " | ".join(cells) + " |")


def _add_audio_setting_table(
    lines: list[str], runs: dict, models: list[str]
) -> None:
    """Accuracy by audio setting (single vs double)."""
    # Check if any run has audio setting data
    has_data = False
    for key, run in runs.items():
        m = run["metrics"]
        if m.get("confusion_matrix_single") or m.get("confusion_matrix_double"):
            has_data = True
            break

    if not has_data:
        return

    lines += [
        "## Accuracy by Audio Setting",
        "",
        "Single-audio samples provide only the student recording. "
        "Double-audio samples include both a reference and student recording.",
        "",
    ]

    model_names = _model_display_names(runs, models)
    header = "| Task | Setting | " + " | ".join(model_names) + " |"
    sep = "|------|---------|" + "|".join(["---:" for _ in models]) + "|"
    lines += [header, sep]

    for task in ["mcq", "pairwise"]:
        display_task = task.replace("_", " ").title()
        first_row = True
        for setting, cm_key in [
            ("Single", "confusion_matrix_single"),
            ("Double", "confusion_matrix_double"),
        ]:
            values = []
            for model in models:
                run = _find_run(runs, task, model)
                if run is None:
                    values.append(None)
                else:
                    values.append(_accuracy_from_cm(run["metrics"].get(cm_key, {})))

            # Skip rows where all values are None
            if all(v is None for v in values):
                continue

            task_label = f"**{display_task}**" if first_row else ""
            first_row = False

            best_idx = _find_best(values, higher_better=True)
            cells = []
            for j, v in enumerate(values):
                formatted = _fmt_3f(v) if v is not None else "—"
                if j == best_idx:
                    formatted = f"**{formatted}**"
                cells.append(formatted)

            lines.append(f"| {task_label} | {setting} | " + " | ".join(cells) + " |")

    lines += ["", ""]


def _add_judge_table(
    lines: list[str], runs: dict, models: list[str]
) -> None:
    """LLM-as-a-Judge results for open-ended task."""
    # Check if any open_ended run has judge data
    has_judge = any(
        run.get("judge_metrics") is not None
        for key, run in runs.items()
        if key[0] == "open_ended"
    )
    if not has_judge:
        return

    lines += [
        "## LLM-as-a-Judge (Open-Ended)",
        "",
        "Quality assessment by a secondary LLM. Samples are first routed based on "
        "content presence before reaching the judge:",
        "",
        "- **No mistake**: neither the model nor the ground truth contains a mistake — nothing to judge.",
        "- **False negative**: the ground truth has a mistake but the model produced none.",
        "- **False positive**: the model reported a mistake but the ground truth has none.",
        "- **LLM evaluated**: both sides have content — sent to the judge for quality rating.",
        "",
        "Routing rates are normalized by total samples. "
        "Quality rates are normalized by the number of successfully judged samples.",
        "",
    ]

    model_names = _model_display_names(runs, models)
    header = "| Metric | " + " | ".join(model_names) + " |"
    sep = "|--------|" + "|".join(["---:" for _ in models]) + "|"
    lines += [header, sep]

    judge_rows = [
        ("No mistake rate", lambda jm: jm.get("no_mistake_rate"), _fmt_pct, False),
        ("False negative rate ↓", lambda jm: jm.get("missed_rate"), _fmt_pct, True),
        ("False positive rate ↓", lambda jm: jm.get("hallucinated_rate"), _fmt_pct, True),
        ("LLM evaluated rate", lambda jm: jm.get("llm_evaluated_rate"), _fmt_pct, False),
        ("Judged", lambda jm: jm.get("judged"), _fmt_int, False),
        ("Parse errors", lambda jm: jm.get("parse_errors"), _fmt_int, False),
        ("—", None, None, False),  # separator row
        ("Correct mistake ↑", lambda jm: jm.get("mistake_rates", {}).get("correct"), _fmt_pct, True),
        ("Helpful feedback ↑", lambda jm: jm.get("feedback_rates", {}).get("helpful"), _fmt_pct, True),
    ]

    for metric_name, extractor, formatter, bold_best in judge_rows:
        if extractor is None:
            # Separator row
            cells = ["—" for _ in models]
            lines.append(f"| | " + " | ".join(cells) + " |")
            continue

        values = []
        for model in models:
            run = _find_run(runs, "open_ended", model)
            jm = run.get("judge_metrics") if run else None
            if jm is None:
                values.append(None)
            else:
                values.append(extractor(jm))

        is_higher = "↑" in metric_name
        is_lower = "↓" in metric_name
        best_idx = _find_best(values, higher_better=is_higher) if bold_best and (is_higher or is_lower) else None

        cells = []
        for j, v in enumerate(values):
            formatted = formatter(v) if v is not None else "—"
            if j == best_idx:
                formatted = f"**{formatted}**"
            cells.append(formatted)

        lines.append(f"| {metric_name} | " + " | ".join(cells) + " |")

    lines += ["", ""]


def _add_per_category_tables(
    lines: list[str], runs: dict, models: list[str]
) -> None:
    """Per-category accuracy tables for each task."""
    model_names = _model_display_names(runs, models)

    for task in TASKS:
        # Collect all categories across models
        categories: set[str] = set()
        for model in models:
            run = _find_run(runs, task, model)
            if run is None:
                continue
            cats = run["metrics"].get("per_category_accuracy", {})
            categories.update(cats.keys())

        if not categories:
            continue

        display_task = task.replace("_", " ").title()
        lines += [
            f"## Per-Category Accuracy — {display_task}",
            "",
        ]

        header = "| Category | " + " | ".join(model_names) + " |"
        sep = "|----------|" + "|".join(["---:" for _ in models]) + "|"
        lines += [header, sep]

        # Sort categories by average accuracy across models (descending)
        cat_avg: dict[str, float] = {}
        for cat in categories:
            accs = []
            for model in models:
                run = _find_run(runs, task, model)
                if run is None:
                    continue
                cat_data = run["metrics"].get("per_category_accuracy", {}).get(cat)
                if cat_data:
                    accs.append(cat_data.get("accuracy", 0))
            cat_avg[cat] = sum(accs) / len(accs) if accs else 0

        for cat in sorted(categories, key=lambda c: -cat_avg[c]):
            values = []
            for model in models:
                run = _find_run(runs, task, model)
                if run is None:
                    values.append(None)
                    continue
                cat_data = run["metrics"].get("per_category_accuracy", {}).get(cat)
                if cat_data:
                    values.append(cat_data.get("accuracy"))
                else:
                    values.append(None)

            best_idx = _find_best(values, higher_better=True)
            cells = []
            for j, v in enumerate(values):
                formatted = _fmt_3f(v) if v is not None else "—"
                if j == best_idx:
                    formatted = f"**{formatted}**"
                cells.append(formatted)

            lines.append(f"| {cat} | " + " | ".join(cells) + " |")

        lines += ["", ""]


def _find_run(runs: dict, task: str, model: str) -> dict | None:
    """Find run by task and model name."""
    for key, run in runs.items():
        if key[0] == task and run["model"] == model:
            return run
    return None


def _model_display_names(runs: dict, models: list[str]) -> list[str]:
    """Short display names for models."""
    return [m.split("/")[-1] if "/" in m else m for m in models]


def _find_best(values: list, higher_better: bool = True) -> int | None:
    """Find index of best value. Returns None if all None."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    if higher_better:
        return max(valid, key=lambda x: x[1])[0]
    else:
        return min(valid, key=lambda x: x[1])[0]


def _accuracy_from_cm(cm_dict: dict) -> float | None:
    if not cm_dict:
        return None
    correct = sum(v for k, v in cm_dict.items() if k.split("|")[0] == k.split("|")[1])
    total = sum(cm_dict.values())
    return correct / total if total > 0 else None


def _fmt_pct(v) -> str:
    return f"{v:.1%}" if v is not None else "—"


def _fmt_3f(v) -> str:
    return f"{v:.3f}" if v is not None else "—"


def _fmt_int(v) -> str:
    return str(int(v)) if v is not None else "—"


if __name__ == "__main__":
    main()
