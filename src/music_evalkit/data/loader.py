from pathlib import Path

import pandas as pd

from music_evalkit.data.schema import (
    InferenceSample,
    MCQInference,
    OpenEndedInference,
    PairwiseInference,
    TaskType,
)

_TASK_DIRS = {
    TaskType.OPEN_ENDED: "open_ended",
    TaskType.MCQ: "mcq",
    TaskType.PAIRWISE: "pairwise",
}

_TASK_MODELS: dict[TaskType, type[InferenceSample]] = {
    TaskType.OPEN_ENDED: OpenEndedInference,
    TaskType.MCQ: MCQInference,
    TaskType.PAIRWISE: PairwiseInference,
}


class DatasetLoader:
    """Loads processed CSV data into Pydantic inference models."""

    @classmethod
    def load_task(
        cls,
        task_type: TaskType,
        data_dir: Path | str = Path("data/processed"),
    ) -> list[InferenceSample]:
        """Load samples for a specific task type from CSV.

        Args:
            task_type: The task type to load (OPEN_ENDED, MCQ, or PAIRWISE).
            data_dir: Base directory containing processed data.

        Returns:
            List of validated Pydantic model instances.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValidationError: If any row fails Pydantic validation.
        """
        data_dir = Path(data_dir)
        task_dir = _TASK_DIRS[task_type]
        csv_path = data_dir / task_dir / "samples.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        model_class = _TASK_MODELS[task_type]

        samples = []
        for _, row in df.iterrows():
            # Convert row to dict, handling NaN values
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            sample = model_class(**row_dict)
            samples.append(sample)

        return samples

    @classmethod
    def load_all(
        cls,
        data_dir: Path | str = Path("data/processed"),
    ) -> dict[TaskType, list[InferenceSample]]:
        """Load all task types from the data directory.

        Args:
            data_dir: Base directory containing processed data.

        Returns:
            Dictionary mapping TaskType to list of samples.
            Only includes task types that have data files.
        """
        data_dir = Path(data_dir)
        result = {}

        for task_type in [TaskType.OPEN_ENDED, TaskType.MCQ, TaskType.PAIRWISE]:
            task_dir = _TASK_DIRS[task_type]
            csv_path = data_dir / task_dir / "samples.csv"

            if csv_path.exists():
                result[task_type] = cls.load_task(task_type, data_dir)

        return result
