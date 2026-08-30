from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel

from music_evalkit.messages.types import Message


class LLMConfig(BaseModel):
    """Configuration for LLM client."""

    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    supports_audio: bool = False  # Whether this provider config supports audio
    json_mode: bool = False  # Whether to request JSON output via response_format
    # HuggingFace-specific
    device: Optional[str] = None
    torch_dtype: Optional[str] = None

class LLMResponse(BaseModel):
    """Standardized response from any provider."""

    text: str
    model: str
    usage: Optional[dict] = None
    raw_response: Optional[dict] = None


class BaseLLMClient(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    def supports_audio(self) -> bool:
        """Whether this provider supports audio input."""
        return self.config.supports_audio

    @abstractmethod
    def generate(self, messages: list[Message]) -> LLMResponse:
        """Generate response from messages.

        Args:
            messages: List of messages in OpenAI-compatible format.

        Returns:
            LLMResponse with the generated text and metadata.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (API key set, model loaded, etc.)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name})"
