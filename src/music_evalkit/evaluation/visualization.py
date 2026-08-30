from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from music_evalkit.evaluation.evaluator import EvaluationMetrics
from music_evalkit.evaluation.utils import cm_dict_to_dataframe


def plot_confusion_matrix(
    cm_dict: dict,
    title: str,
    output_path: Path,
    normalize: bool = False,
) -> None:
    """Plot a confusion matrix as a heatmap and save to file.

    Args:
        cm_dict: Dict with "actual|predicted" keys and count values.
        title: Plot title.
        output_path: Path to save the PNG file.
        normalize: If True, normalize rows (for recall analysis).
    """
    df = cm_dict_to_dataframe(cm_dict)
    if df.empty:
        return

    if normalize:
        row_sums = df.sum(axis=1)
        df = df.div(row_sums, axis=0).fillna(0)
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        fmt = "d"
        vmin, vmax = None, None

    # Scale figure size based on number of labels
    n_labels = len(df)
    fig_size = max(6, n_labels * 0.8 + 2)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        df,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.8,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_confusion_matrices(
    metrics: EvaluationMetrics,
    task_name: str,
    output_dir: Path,
) -> list[Path]:
    """Generate all confusion matrix heatmaps for a task.

    Args:
        metrics: Evaluation metrics containing confusion matrix dicts.
        task_name: Task name (used in filenames and titles).
        output_dir: Directory to save PNG files.

    Returns:
        List of paths to generated PNG files.
    """
    generated = []
    display_name = task_name.replace("_", " ").title()

    matrix_configs = [
        (metrics.confusion_matrix, "combined", ""),
        (metrics.confusion_matrix_single, "single", " (Single Audio)"),
        (metrics.confusion_matrix_double, "double", " (Double Audio)"),
    ]

    for cm_dict, suffix, label in matrix_configs:
        if not cm_dict:
            continue

        # Raw counts
        filename = f"{task_name}_confusion_matrix{'_' + suffix if suffix != 'combined' else ''}.png"
        path = output_dir / filename
        plot_confusion_matrix(
            cm_dict,
            title=f"{display_name} Confusion Matrix{label}",
            output_path=path,
        )
        generated.append(path)

        # Normalized version
        norm_filename = f"{task_name}_confusion_matrix{'_' + suffix if suffix != 'combined' else ''}_normalized.png"
        norm_path = output_dir / norm_filename
        plot_confusion_matrix(
            cm_dict,
            title=f"{display_name} Confusion Matrix{label} (Normalized)",
            output_path=norm_path,
            normalize=True,
        )
        generated.append(norm_path)

    return generated


def plot_confusion_matrix_grid(
    models: list[tuple[str, dict]],
    title: str,
    output_path: Path,
    normalize: bool = True,
) -> None:
    """Plot a 2x2 grid of confusion matrices for multiple models.

    Args:
        models: List of (model_display_name, cm_dict) tuples (up to 4).
        title: Overall figure title.
        output_path: Path to save the PNG file.
        normalize: If True, normalize rows (recall-style).
    """
    n = len(models)
    if n == 0:
        return

    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 6))
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    for idx, (model_name, cm_dict) in enumerate(models):
        ax = axes[idx]
        df = cm_dict_to_dataframe(cm_dict)
        if df.empty:
            ax.set_visible(False)
            continue

        if normalize:
            row_sums = df.sum(axis=1)
            df = df.div(row_sums, axis=0).fillna(0)
            fmt = ".2f"
            vmin, vmax = 0.0, 1.0
        else:
            fmt = "d"
            vmin, vmax = None, None

        sns.heatmap(
            df,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.8,
            linecolor="white",
            ax=ax,
            cbar_kws={"shrink": 0.7},
            annot_kws={"size": 8, "weight": "bold"},
        )
        ax.set_title(model_name, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


_MODEL_DISPLAY_NAMES = {
    "gemini-2.0-flash": "Gemini",
    "nvidia/audio-flamingo-3-hf": "AF3",
    "nvidia/music-flamingo-hf": "MF",
    "qwen2.5-omni-7b": "Qwen",
    "gemini": "Gemini",
    "audio_flamingo": "AF3",
    "music_flamingo": "MF",
    "qwen": "Qwen",
}


def plot_judge_distributions(
    judge_metrics_per_model: dict[str, dict],
    output_path: Path,
) -> None:
    """Plot stacked bar charts of LLM judge mistake and feedback distributions.

    Args:
        judge_metrics_per_model: Dict mapping model display name to judge_metrics dict.
        output_path: Path to save the PNG file.
    """
    models = list(judge_metrics_per_model.keys())
    display_names = [_MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    n_judged = [judge_metrics_per_model[m]["judged"] for m in models]

    mistake_categories = ["correct", "partially_correct", "incorrect"]
    feedback_categories = ["helpful", "generic", "unhelpful"]
    mistake_colors = ["#A8CFE8", "#6BAED6", "#2171B5"]
    feedback_colors = ["#A5D5A0", "#74C476", "#238B45"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, categories, colors, dist_key, title in [
        (axes[0], mistake_categories, mistake_colors, "mistake_rates", "Mistake Description Quality"),
        (axes[1], feedback_categories, feedback_colors, "feedback_rates", "Feedback Quality"),
    ]:
        bottoms = np.zeros(len(models))
        for cat, color in zip(categories, colors):
            values = [
                judge_metrics_per_model[m][dist_key].get(cat, 0) * 100
                for m in models
            ]
            bars = ax.bar(display_names, values, bottom=bottoms, color=color, label=cat.replace("_", " ").capitalize(), width=0.5)
            for bar, val in zip(bars, values):
                if val > 4:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%",
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold",
                    )
            bottoms += np.array(values)

        # Annotate n= above each bar
        for i, (name, n) in enumerate(zip(display_names, n_judged)):
            ax.text(i, 102, f"n={n}", ha="center", va="bottom", fontsize=8, color="gray")

        ax.set_title(title, fontsize=13)
        ax.set_ylabel("% of judged samples")
        ax.set_ylim(0, 115)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("LLM-as-a-Judge Quality Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
