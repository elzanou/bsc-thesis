from music_evalkit.messages.types import Message
from music_evalkit.models.base import BaseLLMClient, LLMConfig, LLMResponse
from music_evalkit.models.factory import register_client


@register_client("noop")
class NoopClient(BaseLLMClient):
    """Mock client that returns predefined responses for testing."""

    def __init__(self, config: LLMConfig, mock_response: str = "mock response"):
        super().__init__(config)
        self.mock_response = mock_response
        self.call_count = 0
        self.calls: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> LLMResponse:
        """Return mock response.

        Args:
            messages: List of messages in OpenAI-compatible format.

        Returns:
            Mock LLMResponse with configured mock_response text.
        """
        self.call_count += 1
        self.calls.append(messages)

        return LLMResponse(
            text=self.mock_response,
            model="noop",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def is_available(self) -> bool:
        """NOOP client is always available."""
        return True
