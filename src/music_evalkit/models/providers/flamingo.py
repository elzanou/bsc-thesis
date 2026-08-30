import base64
import tempfile
from pathlib import Path

import torch

from music_evalkit.messages.types import Message
from music_evalkit.models.base import BaseLLMClient, LLMConfig, LLMResponse
from music_evalkit.models.factory import register_client


@register_client("audio_flamingo")
@register_client("music_flamingo")
class FlamingoClient(BaseLLMClient):
    """Client for NVIDIA Flamingo models (Audio Flamingo 3, Music Flamingo).

    Both models use AudioFlamingo3ForConditionalGeneration and share the same
    inference pattern. They differ in specialization:
    - Audio Flamingo 3: general audio understanding (speech, sounds, music)
    - Music Flamingo: music-specialized (genre, tempo, key, instruments)

    Usage:
        # Audio Flamingo 3
        config = LLMConfig(
            model_name="nvidia/audio-flamingo-3-hf",
            device="cuda",
            torch_dtype="bfloat16",
        )

        # Music Flamingo
        config = LLMConfig(
            model_name="nvidia/music-flamingo-hf",
            device="cuda",
            torch_dtype="bfloat16",
        )
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_dtype = self._parse_dtype(config.torch_dtype)
        self._attn_impl: str | None = None  # Cache attention implementation

    def _parse_dtype(self, dtype_str: str | None) -> torch.dtype:
        """Parse dtype string to torch dtype."""
        if dtype_str is None:
            return torch.bfloat16  # Default for Flamingo models
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def _load_model(self) -> None:
        """Lazy-load model and processor."""
        if self._model is not None:
            return

        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        model_name = self.config.model_name

        self._processor = AutoProcessor.from_pretrained(model_name)

        # Use flash_attention_2 if available for faster inference
        attn_impl = self._get_attn_implementation()

        self._model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self._torch_dtype,
            device_map=self._device,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )

    def _get_attn_implementation(self) -> str:
        """Get best available attention implementation (cached)."""
        if self._attn_impl is not None:
            return self._attn_impl

        try:
            from flash_attn.flash_attn_interface import flash_attn_func  # noqa: F401

            print("Using flash_attention_2 for faster inference")
            self._attn_impl = "flash_attention_2"
        except Exception:
            print("flash-attn not available, using default attention (sdpa)")
            self._attn_impl = "sdpa"

        return self._attn_impl

    def _to_flamingo_conversation(
        self, messages: list[Message]
    ) -> tuple[list[dict], list[Path]]:
        """Convert OpenAI-compatible messages to Flamingo conversation format.

        Flamingo expects audio as file paths, so we write base64 audio to temp files.
        The chat template supports system/user/assistant roles natively.

        Returns:
            Tuple of (conversation list, list of temp file paths to clean up).
        """
        conversation = []
        temp_files = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                conversation.append(
                    {"role": role, "content": [{"type": "text", "text": content}]}
                )
                continue

            flamingo_content = []
            for part in content:
                if part["type"] == "text":
                    flamingo_content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "input_audio":
                    audio_info = part["input_audio"]
                    if "path" in audio_info:
                        flamingo_content.append({"type": "audio", "path": audio_info["path"]})
                    else:
                        audio_bytes = base64.b64decode(audio_info["data"])
                        audio_format = audio_info.get("format", "wav")
                        temp_file = tempfile.NamedTemporaryFile(
                            suffix=f".{audio_format}", delete=False
                        )
                        temp_file.write(audio_bytes)
                        temp_file.close()
                        temp_files.append(Path(temp_file.name))
                        flamingo_content.append({"type": "audio", "path": temp_file.name})

            conversation.append({"role": role, "content": flamingo_content})

        return conversation, temp_files

    def generate(self, messages: list[Message]) -> LLMResponse:
        """Generate response from OpenAI-compatible messages."""
        self._load_model()

        # Convert to Flamingo format (system role supported natively by chat template)
        conversation, temp_files = self._to_flamingo_conversation(messages)

        # Inject "{" prefix so the model continues the response as JSON.
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": "{"}]})

        try:
            inputs = self._processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
            ).to(self._model.device)

            # Cast float tensors to model dtype (audio features are float32, model is bfloat16)
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    inputs[key] = value.to(self._torch_dtype)

            # Generate
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens or 1024,
                    temperature=self.config.temperature
                    if self.config.temperature > 0
                    else None,
                    do_sample=self.config.temperature > 0
                    if self.config.temperature
                    else False,
                )

            # Decode response; restore the prefilled "{" prefix
            generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
            response_text = "{" + self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return LLMResponse(
                text=response_text,
                model=self.config.model_name,
                usage={
                    "prompt_tokens": inputs.input_ids.shape[1],
                    "completion_tokens": generated_ids.shape[1],
                    "total_tokens": output_ids.shape[1],
                },
            )

        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def is_available(self) -> bool:
        """Check if model can be loaded."""
        try:
            from transformers import AutoConfig

            AutoConfig.from_pretrained(self.config.model_name)
            return True
        except Exception:
            return False
