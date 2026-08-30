import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class TaskType(str, Enum):
    """Task types for evaluation."""

    OPEN_ENDED = "open_ended"
    MCQ = "mcq"
    PAIRWISE = "pairwise"


class Instrument(str, Enum):
    """Supported instruments."""

    PIANO = "piano"
    GUITAR = "guitar"


class OpenEndedBase(BaseModel):
    """Base fields for open-ended samples."""

    id: str
    instruction: str
    instrument: Instrument
    piece_type: str
    audio_setting: Literal["single", "double"]
    mistake_category: str
    mistake: str | None = None  # None for no_mistake category
    feedback: str | None = None  # None for no_mistake category


class OpenEndedRaw(OpenEndedBase):
    """Raw open-ended sample from CSV.

    Used for validating input data before processing.
    """

    student_audio: str
    audio_ref: str | None = None
    audio_ref_title: str | None = None
    time_interval: str | None = None

    @model_validator(mode="after")
    def validate_double_audio_has_ref(self):
        if self.audio_setting == "double" and not self.audio_ref:
            raise ValueError("audio_setting='double' requires audio_ref")
        return self

    @field_validator("time_interval")
    @classmethod
    def validate_time_interval_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        pattern = r"^\d{2}:\d{2}-\d{2}:\d{2}$"
        if not re.match(pattern, v):
            raise ValueError("time_interval must be MM:SS-MM:SS format")
        return v


class OpenEndedInference(OpenEndedBase):
    """Open-ended sample ready for inference.

    Ground truth for evaluation: mistake, feedback
    """

    audio_path: str


class MCQBase(BaseModel):
    """Base fields for MCQ samples."""

    id: str
    instruction: str
    instrument: Instrument
    piece_type: str
    audio_setting: Literal["single", "double"]
    mistake_category: str
    options: str  # comma-separated
    answer: str

    @field_validator("options")
    @classmethod
    def validate_options_count(cls, v: str) -> str:
        parts = [p.strip() for p in v.split(",")]
        if len(parts) != 4:
            raise ValueError(f"options must have exactly 4 items, got {len(parts)}")
        return v

    @model_validator(mode="after")
    def validate_answer_in_options(self):
        parts = [p.strip() for p in self.options.split(",")]
        if self.answer.strip() not in parts:
            raise ValueError("answer must be one of options")
        return self


class MCQRaw(MCQBase):
    """Raw MCQ sample from CSV.

    Used for validating input data before processing.
    """

    student_audio: str
    audio_ref: str | None = None
    audio_ref_title: str | None = None
    time_interval: str | None = None

    @model_validator(mode="after")
    def validate_double_audio_has_ref(self):
        if self.audio_setting == "double" and not self.audio_ref:
            raise ValueError("audio_setting='double' requires audio_ref")
        return self

    @field_validator("time_interval")
    @classmethod
    def validate_time_interval_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        pattern = r"^\d{2}:\d{2}-\d{2}:\d{2}$"
        if not re.match(pattern, v):
            raise ValueError("time_interval must be MM:SS-MM:SS format")
        return v


class MCQInference(MCQBase):
    """MCQ sample ready for inference.

    Ground truth for evaluation: answer
    """

    audio_path: str


class PairwiseBase(BaseModel):
    """Base fields for pairwise samples."""

    id: str
    question: str
    instrument: Instrument
    category: str
    label: str

    @field_validator("label", mode="before")
    @classmethod
    def validate_label_value(cls, v) -> str:
        v_str = str(v).strip().upper()
        # Accept legacy 1/2 and map to A/B
        if v_str == "1":
            return "A"
        if v_str == "2":
            return "B"
        if v_str not in ("A", "B"):
            raise ValueError(f"label must be A or B, got {v!r}")
        return v_str


class PairwiseRaw(PairwiseBase):
    """Raw pairwise sample from CSV.

    Compares two student audio recordings.
    """

    audio_1: str
    audio_2: str


class PairwiseInference(PairwiseBase):
    """Pairwise sample ready for inference.

    Ground truth for evaluation: label (A or B)
    """

    audio_path: str


RawSample = OpenEndedRaw | MCQRaw | PairwiseRaw
InferenceSample = OpenEndedInference | MCQInference | PairwiseInference
