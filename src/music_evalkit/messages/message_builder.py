import base64
from pathlib import Path

from music_evalkit.data.schema import (
    InferenceSample,
    MCQInference,
    OpenEndedInference,
    PairwiseInference,
    TaskType,
)
from music_evalkit.messages.types import AudioContent, Message, TextContent
from music_evalkit.prompts.templates import get_template


class MessageBuilder:
    """Builds OpenAI-compatible messages from samples."""

    def build(self, sample: InferenceSample, task_type: TaskType | None = None) -> list[Message]:
        """Build messages from a sample.

        Args:
            sample: The evaluation sample.
            task_type: Override task type.

        Returns:
            List of messages in OpenAI-compatible format.
            - System message: task instructions (from template)
            - User message: instruction text + audio content part
        """
        if task_type is None:
            task_type = self._get_task_type(sample)
        audio_setting = getattr(sample, "audio_setting", "single")
        system_prompt = get_template(task_type, audio_setting)
        user_prompt = self._build_user_prompt(sample)

        user_content: list[TextContent | AudioContent] = [
            {"type": "text", "text": user_prompt},
            self._create_audio_content(Path(sample.audio_path)),
        ]

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _get_task_type(self, sample: InferenceSample) -> TaskType:
        """Get task type from sample type."""
        if isinstance(sample, MCQInference):
            return TaskType.MCQ
        elif isinstance(sample, PairwiseInference):
            return TaskType.PAIRWISE
        elif isinstance(sample, OpenEndedInference):
            return TaskType.OPEN_ENDED
        else:
            raise ValueError(f"Unknown sample type: {type(sample)}")

    def _build_user_prompt(self, sample: InferenceSample) -> str:
        """Build user prompt with instruction and options if applicable."""
        if isinstance(sample, PairwiseInference):
            return f"Question: {sample.question}"

        instruction = f"{sample.instruction.rstrip('.')} on {sample.instrument.value}."

        if isinstance(sample, MCQInference):
            choices = [opt.strip() for opt in sample.options.split(",")]
            return (
                f"Instruction: {instruction}\n\n"
                f"A) {choices[0]}\n"
                f"B) {choices[1]}\n"
                f"C) {choices[2]}\n"
                f"D) {choices[3]}"
            )
        return f"Instruction: {instruction}"

    def _create_audio_content(self, audio_path: Path) -> AudioContent:
        """Create audio content from a file path."""
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(audio_bytes).decode("utf-8"),
                "format": audio_path.suffix.lower().lstrip("."),
            },
        }
