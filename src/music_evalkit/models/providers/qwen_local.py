import base64
import tempfile
from pathlib import Path

import torch

from music_evalkit.messages.types import Message
from music_evalkit.models.base import BaseLLMClient, LLMConfig, LLMResponse
from music_evalkit.models.factory import register_client


@register_client("qwen_local")
class QwenLocalClient(BaseLLMClient):
    """Client for Qwen2.5-Omni running locally on GPU (e.g. RunPod).

    Usage:
        config = LLMConfig(
            model_name="Qwen/Qwen2.5-Omni-7B",
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

    def _parse_dtype(self, dtype_str: str | None) -> torch.dtype:
        """Parse dtype string to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str or "bfloat16", torch.bfloat16)

    def _load_model(self) -> None:
        """Lazy-load model and processor."""
        if self._model is not None:
            return

        from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor

        model_name = self.config.model_name

        self._processor = AutoProcessor.from_pretrained(model_name)

        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self._torch_dtype,
            device_map=self._device,
            low_cpu_mem_usage=True,
        )

    def _to_qwen_conversation(
        self, messages: list[Message]
    ) -> tuple[list[dict], list, list[Path]]:
        """Convert OpenAI-compatible messages to Qwen2.5-Omni conversation format.

        Returns:
            Tuple of (conversation list, list of audio arrays, temp files to clean up).
        """
        import librosa

        conversation = []
        audios = []
        temp_files = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                conversation.append({"role": role, "content": content})
                continue

            qwen_content = []
            for part in content:
                if part["type"] == "text":
                    qwen_content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "input_audio":
                    audio_info = part["input_audio"]

                    if "path" in audio_info:
                        audio_path = audio_info["path"]
                    else:
                        audio_data = audio_info["data"]
                        audio_format = audio_info.get("format", "wav")
                        audio_bytes = base64.b64decode(audio_data)
                        temp_file = tempfile.NamedTemporaryFile(
                            suffix=f".{audio_format}", delete=False
                        )
                        temp_file.write(audio_bytes)
                        temp_file.close()
                        temp_files.append(Path(temp_file.name))
                        audio_path = temp_file.name

                    sr = self._processor.feature_extractor.sampling_rate
                    audio_array, _ = librosa.load(audio_path, sr=sr)
                    audios.append(audio_array)

                    qwen_content.append({"type": "audio", "audio_url": audio_path})

            conversation.append({"role": role, "content": qwen_content})

        return conversation, audios, temp_files

    def generate(self, messages: list[Message]) -> LLMResponse:
        """Generate response from OpenAI-compatible messages."""
        self._load_model()

        conversation, audios, temp_files = self._to_qwen_conversation(messages)

        # Inject "{" prefix so the model continues the response as JSON.
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": "{"}]})

        try:
            text = self._processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )

            inputs = self._processor(
                text=text,
                audios=audios if audios else None,
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    inputs[key] = value.to(self._torch_dtype)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens or 1024,
                    temperature=self.config.temperature if self.config.temperature > 0 else None,
                    do_sample=self.config.temperature > 0 if self.config.temperature else False,
                )

            # Decode response; restore the prefilled "{" prefix
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
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
