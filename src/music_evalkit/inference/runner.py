import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from music_evalkit.data.schema import InferenceSample, TaskType
from music_evalkit.inference.cache import InferenceCache
from music_evalkit.messages.message_builder import MessageBuilder
from music_evalkit.models.base import BaseLLMClient, LLMResponse
from music_evalkit.models.config import InferenceConfig


@dataclass
class InferenceResult:
    """Result of inference on a single sample."""

    sample_id: str
    task_type: str
    response_text: str
    model: str
    cached: bool = False
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone(timedelta(hours=2))).isoformat())


@dataclass
class InferenceRunResult:
    """Result of running inference on a dataset."""

    results: list[InferenceResult] = field(default_factory=list)
    total: int = 0
    cached: int = 0
    failed: int = 0
    model: str = ""


class InferenceRunner:
    """Orchestrates inference across samples with caching and retry."""

    def __init__(
        self,
        client: BaseLLMClient,
        message_builder: MessageBuilder,
        cache: Optional[InferenceCache] = None,
        config: Optional[InferenceConfig] = None,
    ):
        """Initialize runner.

        Args:
            client: LLM client to use for inference.
            message_builder: Builder for constructing messages from samples.
            cache: Optional cache for storing results.
            config: Inference configuration.
        """
        self.client = client
        self.message_builder = message_builder
        self.cache = cache
        self.config = config or InferenceConfig()

    def _run_sample(
        self,
        sample: InferenceSample,
        task_type: TaskType,
    ) -> InferenceResult:
        """Run inference on single sample with retry logic.

        Args:
            sample: Sample to process.
            task_type: Task type for prompt formatting.

        Returns:
            InferenceResult with response or error.
        """
        model = self.client.config.model_name

        # Build messages
        messages = self.message_builder.build(sample, task_type)

        # Check cache
        cache_key = None
        if self.cache:
            cache_key = self.cache.make_key(model, str(messages), [Path(sample.audio_path)])
            cached_response = self.cache.get(cache_key, model)
            if cached_response:
                return InferenceResult(
                    sample_id=sample.id,
                    task_type=task_type.value,
                    response_text=cached_response.text,
                    model=model,
                    cached=True,
                )

        # Run with retry
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                response: LLMResponse = self.client.generate(messages)

                # Cache successful response
                if self.cache and cache_key:
                    self.cache.set(cache_key, model, response)

                return InferenceResult(
                    sample_id=sample.id,
                    task_type=task_type.value,
                    response_text=response.text,
                    model=model,
                    cached=False,
                )

            except Exception as e:
                if self._should_skip(e):
                    return InferenceResult(
                        sample_id=sample.id,
                        task_type=task_type.value,
                        response_text="",
                        model=model,
                        error=f"Skipped: {e}",
                    )
                if not self._should_retry(e):
                    raise
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    time.sleep(delay)

        # All retries exhausted
        raise last_error

    def run(
        self,
        samples: list[InferenceSample],
        task_type: TaskType,
        output_path: Optional[Path] = None,
    ) -> InferenceRunResult:
        """Run inference on samples.

        Args:
            samples: List of samples to process.
            task_type: Task type for prompt formatting.
            output_path: Optional path to save results incrementally as JSONL.

        Returns:
            InferenceRunResult with all results.
        """
        results = []
        cached_count = 0
        failed_count = 0

        # Use context manager for file handling
        output_file = None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = open(output_path, "w")

        try:
            for sample in tqdm(samples, desc="Inference"):
                result = self._run_sample(sample, task_type)
                results.append(result)

                if result.cached:
                    cached_count += 1
                if result.error:
                    failed_count += 1

                # Write result to file
                if output_file:
                    result_dict = {
                        "sample_id": result.sample_id,
                        "task_type": result.task_type,
                        "response_text": result.response_text,
                        "model": result.model,
                        "cached": result.cached,
                        "error": result.error,
                        "timestamp": result.timestamp,
                    }
                    output_file.write(json.dumps(result_dict) + "\n")
                    output_file.flush()
        finally:
            if output_file:
                output_file.close()

        return InferenceRunResult(
            results=results,
            total=len(results),
            cached=cached_count,
            failed=failed_count,
            model=self.client.config.model_name,
        )

    def _should_skip(self, error: Exception) -> bool:
        """Determine if error means this sample should be skipped (not retried, not fatal).

        Covers cases like provider payload limits that only affect specific samples.
        """
        error_str = str(error).lower()
        skip_patterns = [
            "max bytes",
            "payload too large",
            "request entity too large",
            "413",
        ]
        return any(pattern in error_str for pattern in skip_patterns)

    def _should_retry(self, error: Exception) -> bool:
        """Determine if error is retryable.

        Args:
            error: Exception that occurred.

        Returns:
            True if should retry.
        """
        error_str = str(error).lower()
        retryable_patterns = [
            "rate limit",
            "timeout",
            "connection",
            "temporary",
            "503",
            "429",
            "500",
        ]
        return any(pattern in error_str for pattern in retryable_patterns)
