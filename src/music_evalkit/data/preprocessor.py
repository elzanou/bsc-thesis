"""
Usage:
    python -m music_evalkit.data.preprocess --task open_ended
    python -m music_evalkit.data.preprocess --task mcq
    python -m music_evalkit.data.preprocess --task pairwise
"""

import logging
import random
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from music_evalkit.data.audio import AudioProcessor
from music_evalkit.data.schema import (
    MCQInference,
    MCQRaw,
    OpenEndedInference,
    OpenEndedRaw,
    PairwiseInference,
    PairwiseRaw,
    TaskType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 48000
STUDENT_AUDIO_DIR = Path("data/audios")
YOUTUBE_CACHE_DIR = Path("data/cache/youtube")
OUTPUT_BASE_DIR = Path("data/processed")

# Sheet names in XLSX
SHEET_NAMES = {
    TaskType.OPEN_ENDED: "open_ended",
    TaskType.MCQ: "mcq",
    TaskType.PAIRWISE: "pairwise_target_match",
}


class Preprocessor:
    """Main preprocessing orchestrator."""

    def __init__(
        self,
        task: TaskType,
        xlsx_path: Path,
        output_dir: Path,
        audio_processor: AudioProcessor,
        *,
        reverse_pairwise: bool = False,
    ):
        """Initialize preprocessor.

        Args:
            task: Task type to process
            xlsx_path: Path to XLSX file with annotations
            output_dir: Output directory for processed data
            audio_processor: AudioProcessor instance for audio operations
            reverse_pairwise: Swap audio order and flip labels for pairwise task
        """
        self.task = task
        self.xlsx_path = xlsx_path
        self.output_dir = output_dir
        self.audio_processor = audio_processor
        self.reverse_pairwise = reverse_pairwise
        self.errors: list[dict] = []

    def load_data(self) -> pd.DataFrame:
        """Load data from XLSX.

        Returns:
            DataFrame with annotation data
        """
        sheet_name = SHEET_NAMES[self.task]
        df = pd.read_excel(self.xlsx_path, sheet_name=sheet_name)
        df = df.dropna(how="all")
        logger.info(f"Loaded {len(df)} rows from sheet '{sheet_name}'")
        return df

    def process_open_ended(self, df: pd.DataFrame) -> list[OpenEndedInference]:
        """Process open_ended task samples.

        Args:
            df: DataFrame with open_ended annotations

        Returns:
            List of inference-ready samples
        """
        samples = []
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                # Parse raw sample
                raw = OpenEndedRaw(
                    id=str(row["id"]),
                    instruction=row["instruction"],
                    instrument=row["instrument"].lower(),
                    piece_type=row["piece_type"],
                    audio_setting=row["audio_setting"],
                    mistake_category=row["mistake_category"],
                    mistake=row["mistake"] if pd.notna(row["mistake"]) else None,
                    feedback=row["feedback"] if pd.notna(row["feedback"]) else None,
                    student_audio=row["student_audio"],
                    audio_ref=row["audio_ref"] if pd.notna(row["audio_ref"]) else None,
                    audio_ref_title=row["audio_ref_title"] if pd.notna(row.get("audio_ref_title")) else None,
                    time_interval=row["time_interval"] if pd.notna(row.get("time_interval")) else None,
                )

                # Process audio: reference FIRST, student SECOND (separated by beep)
                output_audio_path = audio_dir / f"{raw.id}.wav"
                audio_paths = []

                if raw.audio_setting == "double" and raw.audio_ref:
                    ref_path = self.audio_processor.download_youtube_audio(
                        raw.audio_ref, raw.time_interval
                    )
                    if ref_path is None:
                        raise ValueError(f"Failed to download reference audio: {raw.audio_ref}")
                    audio_paths.append(ref_path)

                student_path = self.audio_processor.get_student_audio_path(raw.student_audio)
                if not student_path.exists():
                    raise FileNotFoundError(f"Student audio not found: {student_path}")
                audio_paths.append(student_path)

                if not self.audio_processor.concatenate_audio(audio_paths, output_audio_path):
                    raise RuntimeError("Audio concatenation failed")

                # Create inference sample
                inference = OpenEndedInference(
                    id=raw.id,
                    instruction=raw.instruction,
                    instrument=raw.instrument,
                    piece_type=raw.piece_type,
                    audio_setting=raw.audio_setting,
                    mistake_category=raw.mistake_category,
                    mistake=raw.mistake,
                    feedback=raw.feedback,
                    audio_path=str(output_audio_path),
                )
                samples.append(inference)

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                self.errors.append({"row": idx, "id": row.get("id"), "error": str(e)})

        return samples

    def process_mcq(self, df: pd.DataFrame) -> list[MCQInference]:
        """Process MCQ task samples.

        Args:
            df: DataFrame with MCQ annotations

        Returns:
            List of inference-ready samples
        """
        samples = []
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                raw = MCQRaw(
                    id=str(row["id"]),
                    instruction=row["instruction"],
                    instrument=row["instrument"].lower(),
                    piece_type=row["piece_type"],
                    audio_setting=row["audio_setting"],
                    mistake_category=row["mistake_category"],
                    options=row["options"],
                    answer=row["answer"],
                    student_audio=row["student_audio"],
                    audio_ref=row["audio_ref"] if pd.notna(row["audio_ref"]) else None,
                    audio_ref_title=row["audio_ref_title"] if pd.notna(row.get("audio_ref_title")) else None,
                    time_interval=row["time_interval"] if pd.notna(row.get("time_interval")) else None,
                )

                # Process audio: reference FIRST, student SECOND (separated by beep)
                output_audio_path = audio_dir / f"{raw.id}.wav"
                audio_paths = []

                if raw.audio_setting == "double" and raw.audio_ref:
                    ref_path = self.audio_processor.download_youtube_audio(
                        raw.audio_ref, raw.time_interval
                    )
                    if ref_path is None:
                        raise ValueError(f"Failed to download reference audio: {raw.audio_ref}")
                    audio_paths.append(ref_path)

                student_path = self.audio_processor.get_student_audio_path(raw.student_audio)
                if not student_path.exists():
                    raise FileNotFoundError(f"Student audio not found: {student_path}")
                audio_paths.append(student_path)

                if not self.audio_processor.concatenate_audio(audio_paths, output_audio_path):
                    raise RuntimeError("Audio concatenation failed")

                # Shuffle options with sample ID as seed (deterministic per sample)
                rng = random.Random(raw.id)
                choices = [opt.strip() for opt in raw.options.split(",")]
                rng.shuffle(choices)
                shuffled_options = ", ".join(choices)

                inference = MCQInference(
                    id=raw.id,
                    instruction=raw.instruction,
                    instrument=raw.instrument,
                    piece_type=raw.piece_type,
                    audio_setting=raw.audio_setting,
                    mistake_category=raw.mistake_category,
                    options=shuffled_options,
                    answer=raw.answer,
                    audio_path=str(output_audio_path),
                )
                samples.append(inference)

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                self.errors.append({"row": idx, "id": row.get("id"), "error": str(e)})

        return samples

    def process_pairwise(self, df: pd.DataFrame) -> list[PairwiseInference]:
        """Process pairwise task samples.

        Args:
            df: DataFrame with pairwise annotations

        Returns:
            List of inference-ready samples
        """
        samples = []
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                raw = PairwiseRaw(
                    id=str(row["id"]),
                    question=row["question"],
                    instrument=row["instrument"].lower(),
                    category=row["category"],
                    label=str(row["label"]),
                    audio_1=row["audio_1"],
                    audio_2=row["audio_2"],
                )

                audio_1_path = self.audio_processor.get_student_audio_path(raw.audio_1)
                audio_2_path = self.audio_processor.get_student_audio_path(raw.audio_2)

                if not audio_1_path.exists():
                    raise FileNotFoundError(f"Audio 1 not found: {audio_1_path}")
                if not audio_2_path.exists():
                    raise FileNotFoundError(f"Audio 2 not found: {audio_2_path}")

                # Beep-concatenated file: Recording A + beep + Recording B
                # When reversed, swap order so Recording B plays first
                if self.reverse_pairwise:
                    audio_order = [audio_2_path, audio_1_path]
                    flipped_label = "B" if raw.label == "A" else "A"  # A→B, B→A
                else:
                    audio_order = [audio_1_path, audio_2_path]
                    flipped_label = raw.label

                output_audio_path = audio_dir / f"{raw.id}.wav"
                if not self.audio_processor.concatenate_audio(
                    audio_order, output_audio_path
                ):
                    raise RuntimeError("Audio concatenation failed")

                inference = PairwiseInference(
                    id=raw.id,
                    question=raw.question,
                    instrument=raw.instrument,
                    category=raw.category,
                    label=flipped_label,
                    audio_path=str(output_audio_path),
                )
                samples.append(inference)

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                self.errors.append({"row": idx, "id": row.get("id"), "error": str(e)})

        return samples

    def save_samples(self, samples: list) -> Path:
        """Save samples to CSV.

        Args:
            samples: List of inference samples

        Returns:
            Path to saved CSV file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "samples.csv"

        data = [s.model_dump() for s in samples]
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(samples)} samples to {output_path}")
        return output_path

    def run(self) -> Path:
        """Run preprocessing pipeline.

        Returns:
            Path to output CSV file
        """
        logger.info(f"Starting preprocessing for task: {self.task.value}")

        df = self.load_data()

        if self.task == TaskType.OPEN_ENDED:
            samples = self.process_open_ended(df)
        elif self.task == TaskType.MCQ:
            samples = self.process_mcq(df)
        elif self.task == TaskType.PAIRWISE:
            samples = self.process_pairwise(df)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        output_path = self.save_samples(samples)

        if self.errors:
            logger.warning(f"Completed with {len(self.errors)} errors:")
            for err in self.errors:
                logger.warning(f"  Row {err['row']} (id={err['id']}): {err['error']}")

        logger.info(f"Preprocessing complete: {len(samples)} samples processed")
        return output_path


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
def main(task: str, xlsx: Path, output_dir: Path | None):
    """Preprocess music evaluation dataset."""
    task_type = TaskType(task)

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
    )

    preprocessor.run()


if __name__ == "__main__":
    main()
