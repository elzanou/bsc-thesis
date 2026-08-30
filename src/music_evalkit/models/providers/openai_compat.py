from openai import OpenAI

from music_evalkit.messages.types import Message
from music_evalkit.models.base import BaseLLMClient, LLMConfig, LLMResponse
from music_evalkit.models.factory import register_client


@register_client("openai_compat")
class OpenAICompatibleClient(BaseLLMClient):
    """Client for any OpenAI-compatible API.

    Usage:
        # OpenAI
        config = LLMConfig(
            model_name="gpt-4o-audio-preview",
            api_key="sk-...",
            supports_audio=True,
        )

        # Gemini via OpenAI-compatible endpoint
        config = LLMConfig(
            model_name="gemini-2.0-flash",
            api_key="your-google-api-key",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            supports_audio=True,
        )

        # Qwen via DashScope OpenAI-compatible endpoint
        config = LLMConfig(
            model_name="qwen2-audio-instruct",
            api_key="your-dashscope-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            supports_audio=True,
        )

        # Ollama (local)
        config = LLMConfig(
            model_name="llama3",
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Ollama doesn't need a real key
        )
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = OpenAI(
            api_key=config.api_key or "",  # Some servers don't need a key
            base_url=config.base_url,
        )

    @property
    def _is_dashscope(self) -> bool:
        return bool(self.config.base_url and "dashscope" in self.config.base_url)

    def _prepare_messages(self, messages: list[Message]) -> list:
        """Add data URI prefix to audio for DashScope compatibility."""
        if not self._is_dashscope:
            return messages
        transformed = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                transformed.append(msg)
                continue
            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_audio":
                    audio = part["input_audio"]
                    new_content.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:;base64,{audio['data']}",
                            "format": audio.get("format", "wav"),
                        },
                    })
                else:
                    new_content.append(part)
            transformed.append({"role": msg["role"], "content": new_content})
        return transformed

    def generate(self, messages: list[Message]) -> LLMResponse:
        """Generate response from messages.

        Args:
            messages: List of messages in OpenAI-compatible format.
                     Can include text and audio content.

        Returns:
            LLMResponse with the generated text and metadata.

        Retries are handled by InferenceRunner, not here.
        """
        messages = self._prepare_messages(messages)

        kwargs = dict(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=self._is_dashscope,
        )
        if self.config.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if self._is_dashscope:
            # DashScope requires streaming for Qwen-Omni
            stream = self._client.chat.completions.create(**kwargs)
            text = ""
            usage = None
            model = self.config.model_name
            response_id = ""
            for chunk in stream:
                response_id = chunk.id or response_id
                model = chunk.model or model
                if chunk.choices and chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }
            return LLMResponse(
                text=text,
                model=model,
                usage=usage,
                raw_response={"id": response_id},
            )

        response = self._client.chat.completions.create(**kwargs)

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=response.model,
            usage=usage,
            raw_response={"id": response.id, "choices": len(response.choices)},
        )

    def is_available(self) -> bool:
        """Check if API is reachable with valid credentials."""
        # For OpenAI-compatible APIs, we check if api_key is set
        # (unless it's a local server like Ollama that doesn't need one)
        if self.config.base_url and "localhost" in self.config.base_url:
            return True
        return bool(self.config.api_key)

