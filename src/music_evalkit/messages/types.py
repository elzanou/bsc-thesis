from typing import Literal, TypedDict, Union


class InputAudio(TypedDict, total=False):
    """Audio data for LLM input.

    For OpenAI-compatible APIs: uses data (base64) + format.
    For local models (Flamingo): can use path directly to avoid encode/decode overhead.
    """

    data: str  # base64-encoded audio (required for OpenAI)
    format: str  # e.g., "wav", "mp3"
    path: str  # original file path (optional, for local models)


class TextContent(TypedDict):
    """Text content in a message."""

    type: Literal["text"]
    text: str


class AudioContent(TypedDict):
    """Audio content in a message."""

    type: Literal["input_audio"]
    input_audio: InputAudio


ContentPart = Union[TextContent, AudioContent]
Content = Union[str, list[ContentPart]]


class Message(TypedDict):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: Content

