#!/usr/bin/env python3
"""
Usage:
    # Evaluate MCQ results
    python scripts/evaluate.py --task mcq --results-dir data/results/mcq/gemini/20260203_111441

    # Evaluate with custom data directory
    python scripts/evaluate.py --task pairwise --results-dir data/results/pairwise/gemini/20260203_111441 --data-dir data/processed

    # Evaluate open-ended with LLM-as-a-Judge
    python scripts/evaluate.py --task open_ended --results-dir data/results/open_ended/gemini/20260228 --judge
    python scripts/evaluate.py --task open_ended --results-dir data/results/open_ended/gemini/20260228 --judge --judge-provider openai

    # Run judge only (skip SBERT evaluation, add judge to existing results)
    python scripts/evaluate.py --task open_ended --results-dir data/results/open_ended/gemini/20260228 --judge-only
"""

import json
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from music_evalkit.evaluation.evaluator import EvaluationMetrics, MCQEvaluator, OpenEndedEvaluator, PairwiseEvaluator
from music_evalkit.evaluation.llm_judge import JudgeMetrics, LLMJudge, run_judge
from music_evalkit.evaluation.report import generate_report
from music_evalkit.evaluation.visualization import plot_all_confusion_matrices
from music_evalkit.evaluation.utils import cm_dict_to_dataframe


TASK_CHOICES = ["mcq", "pairwise", "open_ended"]


@click.command()
@click.option(
    "--task",
    type=click.Choice(TASK_CHOICES),
    required=True,
    help="Task type to evaluate",
)
@click.option(
    "--results-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Run directory containing inference_results.jsonl (e.g., data/results/mcq/gemini/20260203_111441)",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/processed"),
    help="Directory containing ground truth data",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save evaluation results (default: results-dir)",
)
@click.option(
    "--sbert-model",
    type=str,
    default="all-MiniLM-L6-v2",
    help="SBERT model for semantic similarity",
)
@click.option(
    "--mode",
    type=click.Choice(["strict", "recover"]),
    default="recover",
    help="Open-ended parse mode: strict (JSON only) or recover (JSON + fallbacks + category normalisation)",
)
@click.option(
    "--judge",
    is_flag=True,
    default=False,
    help="Run LLM-as-a-Judge for open-ended task (categorical quality assessment)",
)
@click.option(
    "--judge-only",
    is_flag=True,
    default=False,
    help="Run only the judge (skip SBERT evaluation), requires existing evaluation_metrics.json",
)
@click.option(
    "--judge-provider",
    type=click.Choice(["ollama", "openai"]),
    default="ollama",
    help="LLM provider for judging (default: ollama)",
)
@click.option(
    "--judge-model",
    type=str,
    default=None,
    help="Judge model name (default: qwen2.5:7b for ollama, gpt-4o for openai)",
)
@click.option(
    "--judge-resume",
    is_flag=True,
    default=False,
    help="Resume judge from previous run, skipping already-judged samples",
)
@click.option(
    "--judge-output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save judge results (default: output-dir). Use to write new judge results without overwriting existing ones.",
)
def main(
    task: str,
    results_dir: Path,
    data_dir: Path,
    output_dir: Path | None,
    sbert_model: str,
    mode: str,
    judge: bool,
    judge_only: bool,
    judge_provider: str,
    judge_model: str | None,
    judge_resume: bool,
    judge_output_dir: Path | None,
):
    """Evaluate inference results against ground truth."""
    if output_dir is None:
        output_dir = results_dir

    if task == "open_ended":
        output_dir = output_dir / mode

    if judge_only:
        if task != "open_ended":
            click.echo("  Error: --judge-only is only supported for open_ended task", err=True)
            raise SystemExit(1)
        judge = True

    results_path = results_dir / "inference_results.jsonl"
    ground_truth_path = data_dir / task / "samples.csv"

    if not results_path.exists():
        click.echo(f"  Error: {results_path} not found", err=True)
        raise SystemExit(1)
    if not ground_truth_path.exists():
        click.echo(f"  Error: {ground_truth_path} not found", err=True)
        raise SystemExit(1)

    click.echo(f"  Results: {results_path}")
    click.echo(f"  Ground truth: {ground_truth_path}")

    # Run SBERT evaluation (skip in judge-only mode)
    metrics = None
    if not judge_only:
        evaluators = {
            "mcq": MCQEvaluator(),
            "pairwise": PairwiseEvaluator(),
            "open_ended": OpenEndedEvaluator(sbert_model=sbert_model, mode=mode),
        }

        click.echo(f"\n{'='*60}")
        click.echo(f"Evaluating: {task}")
        click.echo(f"{'='*60}")

        sample_results, metrics = evaluators[task].evaluate(
            results_path, ground_truth_path
        )

        _print_metrics(task, metrics)

        _print_confusion_matrix(metrics.confusion_matrix, "Confusion Matrix")
        _print_confusion_matrix(metrics.confusion_matrix_single, "Confusion Matrix (Single Audio)")
        _print_confusion_matrix(metrics.confusion_matrix_double, "Confusion Matrix (Double Audio)")

        _save_results(output_dir, sample_results, metrics)

        generated = plot_all_confusion_matrices(metrics, task, output_dir)
        for path in generated:
            click.echo(f"  Plot: {path}")
    else:
        # Load existing evaluation metrics for report generation
        metrics = _load_existing_metrics(output_dir)

    # Run LLM-as-a-Judge for open-ended
    judge_metrics = None
    if task == "open_ended" and judge:
        click.echo(f"\n{'='*60}")
        click.echo(f"Running LLM-as-a-Judge")
        click.echo(f"{'='*60}")
        effective_judge_output_dir = judge_output_dir or output_dir
        judge_metrics = _run_judge(
            results_path, ground_truth_path, effective_judge_output_dir,
            judge_provider, judge_model, judge_resume, mode=mode,
        )

    # Generate markdown report
    if metrics is not None:
        report_path = generate_report(
            task, metrics, output_dir, results_dir, judge_metrics=judge_metrics
        )
        click.echo(f"  Report: {report_path}")


def _load_existing_metrics(output_dir: Path) -> EvaluationMetrics | None:
    """Load existing evaluation metrics from a previous run."""
    metrics_path = output_dir / "evaluation_metrics.json"
    if not metrics_path.exists():
        click.echo(f"  Warning: {metrics_path} not found, report will be skipped")
        return None
    with open(metrics_path) as f:
        return EvaluationMetrics(**json.load(f))


def _run_judge(
    results_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
    provider: str,
    model: str | None,
    resume: bool,
    mode: str = "recover",
) -> JudgeMetrics:
    """Run LLM-as-a-Judge and return metrics."""
    judge = LLMJudge(provider=provider, model=model)
    display_model = model or ("qwen2.5:7b" if provider == "ollama" else "gpt-4o")
    click.echo(f"  Judge: {provider}/{display_model}")

    def on_progress(done: int, total: int) -> None:
        click.echo(f"  [{done}/{total}]")

    metrics = run_judge(
        judge, results_path, ground_truth_path, output_dir,
        resume=resume, on_progress=on_progress, mode=mode,
    )

    click.echo(f"  Judge results: {output_dir / 'judge_results.jsonl'}")
    click.echo(f"  Judge metrics: {output_dir / 'judge_metrics.json'}")
    _print_judge_summary(metrics)
    return metrics


def _print_judge_summary(metrics: JudgeMetrics) -> None:
    """Print judge summary to console."""
    click.echo(f"\n  Judge Summary:")
    click.echo(f"    Total: {metrics.total}")
    click.echo(f"    Detection: no_mistake={metrics.no_mistake} missed={metrics.missed} hallucinated={metrics.hallucinated}")
    click.echo(f"    LLM evaluated: {metrics.llm_evaluated} (judged: {metrics.judged}, errors: {metrics.parse_errors})")
    if metrics.judged > 0:
        m = metrics.mistake_distribution
        click.echo(f"    Mistake: correct={m.get('correct', 0)} partially_correct={m.get('partially_correct', 0)} incorrect={m.get('incorrect', 0)}")
        fb = metrics.feedback_distribution
        click.echo(f"    Feedback: helpful={fb.get('helpful', 0)} generic={fb.get('generic', 0)} unhelpful={fb.get('unhelpful', 0)}")


def _print_metrics(task_name: str, metrics):
    """Print evaluation metrics to console."""
    click.echo(f"\n  Results:")
    click.echo(f"    Total samples: {metrics.total}")
    click.echo(f"    Parse errors: {metrics.parse_errors}")

    if task_name in ["mcq", "pairwise"]:
        click.echo(f"    Correct: {metrics.correct}")
        click.echo(f"    Accuracy: {metrics.accuracy:.1%}")
        if metrics.precision is not None:
            click.echo(f"    Precision (macro): {metrics.precision:.3f}")
        if metrics.recall is not None:
            click.echo(f"    Recall (macro):    {metrics.recall:.3f}")
        if metrics.f1 is not None:
            click.echo(f"    F1 (macro):        {metrics.f1:.3f}")

        if metrics.per_category_accuracy:
            click.echo(f"\n  Per-category accuracy:")
            for cat, data in sorted(metrics.per_category_accuracy.items()):
                click.echo(f"    {cat}: {data.accuracy:.1%} ({data.correct}/{data.total})")

    elif task_name == "open_ended":
        click.echo(f"    Correct detections: {metrics.correct}")
        click.echo(f"    Detection accuracy: {metrics.accuracy:.1%}")
        if metrics.precision is not None:
            click.echo(f"    Precision (mistake): {metrics.precision:.3f}")
        if metrics.recall is not None:
            click.echo(f"    Recall (mistake):    {metrics.recall:.3f}")
        if metrics.f1 is not None:
            click.echo(f"    F1 (mistake):        {metrics.f1:.3f}")
        if metrics.avg_mistake_similarity is not None:
            click.echo(f"    Avg mistake similarity: {metrics.avg_mistake_similarity:.3f}")
        if metrics.avg_feedback_similarity is not None:
            click.echo(f"    Avg feedback similarity: {metrics.avg_feedback_similarity:.3f}")

        if metrics.per_category_similarity:
            click.echo(f"\n  Per-category avg similarity:")
            for cat, data in sorted(metrics.per_category_similarity.items()):
                click.echo(f"    {cat}: {data.avg_similarity:.3f} ({data.total} samples)")


def _print_confusion_matrix(cm_dict: dict, title: str):
    """Print a confusion matrix as a formatted table."""
    if not cm_dict:
        return

    df = cm_dict_to_dataframe(cm_dict)
    df.index.name = "Actual \\ Predicted"

    click.echo(f"\n  {title}:")
    click.echo(f"  {df.to_string()}")


def _save_results(output_dir: Path, sample_results, metrics):
    """Save evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-sample results
    results_path = output_dir / "evaluation_results.jsonl"
    with open(results_path, "w") as f:
        for result in sample_results:
            f.write(json.dumps(result.model_dump()) + "\n")
    click.echo(f"  Per-sample results: {results_path}")

    # Save metrics
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics.model_dump(), f, indent=2)
    click.echo(f"  Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
