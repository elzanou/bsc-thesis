from music_evalkit.data.schema import TaskType
from music_evalkit.prompts.mcq import MCQ_TEMPLATE, MCQ_TEMPLATE_WITH_REF
from music_evalkit.prompts.open_ended import (
    OPEN_ENDED_BASIC,
    OPEN_ENDED_BASIC_WITH_REF,
)
from music_evalkit.prompts.pairwise import PAIRWISE_TEMPLATE

__all__ = [
    "OPEN_ENDED_BASIC",
    "OPEN_ENDED_BASIC_WITH_REF",
    "MCQ_TEMPLATE",
    "MCQ_TEMPLATE_WITH_REF",
    "PAIRWISE_TEMPLATE",
    "TEMPLATES",
    "get_template",
]

TEMPLATES = {
    TaskType.OPEN_ENDED: OPEN_ENDED_BASIC,
    TaskType.MCQ: MCQ_TEMPLATE,
    TaskType.PAIRWISE: PAIRWISE_TEMPLATE,
}

TEMPLATES_WITH_REF = {
    TaskType.OPEN_ENDED: OPEN_ENDED_BASIC_WITH_REF,
    TaskType.MCQ: MCQ_TEMPLATE_WITH_REF,
}

def get_template(task_type: TaskType, audio_setting: str = "single") -> str:
    """Get a prompt template by task type and audio setting."""
    if audio_setting == "double" and task_type in TEMPLATES_WITH_REF:
        return TEMPLATES_WITH_REF[task_type]
    return TEMPLATES[task_type]
