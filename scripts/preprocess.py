#!/usr/bin/env python3
"""
Usage:
    python scripts/preprocess.py --task open_ended
    python scripts/preprocess.py --task mcq
    python scripts/preprocess.py --task pairwise
"""

from pathlib import Path

import click

from music_evalkit.data.audio import AudioProcessor
from music_evalkit.data.preprocessor import (
    OUTPUT_BASE_DIR,
    SAMPLE_RATE,
    STUDENT_AUDIO_DIR,
    YOUTUBE_CACHE_DIR,
    Preprocessor,
)
from music_evalkit.data.schema import TaskType


@click.command()
@click.option(
    "--task",
    type=click.Choice(["open_ended", "mcq", "pairwise"]),
    required=True,
    help="Task type to preprocess",
)
@click.option(
    "--xlsx",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/Music_samples.xlsx"),
    help="Path to XLSX file",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: data/processed/{task})",
)
@click.option(
    "--reverse",
    is_flag=True,
    default=False,
    help="Pairwise only: swap audio order and flip labels to test positional bias",
)
def main(task: str, xlsx: Path, output_dir: Path | None, reverse: bool):
    """Preprocess music evaluation dataset."""
    task_type = TaskType(task)

    if reverse and task_type != TaskType.PAIRWISE:
        raise click.UsageError("--reverse is only supported for the pairwise task.")

    if output_dir is None:
        output_dir = OUTPUT_BASE_DIR / task

    audio_processor = AudioProcessor(
        student_audio_dir=STUDENT_AUDIO_DIR,
        youtube_cache_dir=YOUTUBE_CACHE_DIR,
        sample_rate=SAMPLE_RATE,
    )

    preprocessor = Preprocessor(
        task=task_type,
        xlsx_path=xlsx,
        output_dir=output_dir,
        audio_processor=audio_processor,
        reverse_pairwise=reverse,
    )

    preprocessor.run()


if __name__ == "__main__":
    main()
