import json
import os
import re
import sys
import time
from enum import StrEnum
from typing import Literal

from openai import (
    OpenAI,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    NotFoundError,
    BadRequestError,
    InternalServerError,
)

from music_evalkit.evaluation.llm_judge.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    JUDGE_RAW_USER_TEMPLATE,
)
from music_evalkit.evaluation.llm_judge.types import (
    FeedbackCategory,
    JudgeResult,
    JudgeSource,
    QualityCategory,
)


_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0

_TRANSIENT_ERRORS = (
    RateLimitError, APIConnectionError, APITimeoutError, InternalServerError,
)
PERMANENT_ERRORS = (
    AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError,
)


class LLMJudge:
    """LLM-as-a-Judge for evaluating open-ended responses.

    Supports:
    - OpenAI API (GPT-4o, GPT-4o-mini)
    - Ollama (local, free) — Qwen2.5, Llama3, Mistral

    Classifies predicted mistake/feedback against ground truth on two
    dimensions:
    - Mistake Description: correct / partially_correct / incorrect
    - Feedback Quality: helpful / generic / unhelpful
    """

    def __init__(
        self,
        provider: Literal["ollama", "openai"] = "ollama",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._provider = provider

        if provider == "ollama":
            self._model = model or "qwen2.5:7b"
            self._base_url = base_url or "http://localhost:11434/v1"
            self._api_key = "ollama"
        elif provider == "openai":
            self._model = model or "gpt-4o"
            self._base_url = base_url
            self._api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self._api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY env var "
                    "or pass api_key."
                )
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Use 'ollama' or 'openai'"
            )

        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Lazy-load OpenAI-compatible client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def judge(
        self,
        pred_reason: str | None,
        pred_mistake: str,
        pred_feedback: str,
        ground_truth_mistake: str,
        ground_truth_feedback: str,
        instruction: str = "",
    ) -> JudgeResult:
        """Judge parsed prediction against ground truth.

        Use for samples where response parsing succeeded. Passes structured
        fields (reasoning, mistake, feedback) without category labels.
        """
        user_prompt = JUDGE_USER_TEMPLATE.format(
            instruction=instruction,
            ground_truth_mistake=ground_truth_mistake,
            ground_truth_feedback=ground_truth_feedback or "",
            pred_reason=pred_reason or "not provided",
            pred_mistake=pred_mistake,
            pred_feedback=pred_feedback or "",
        )
        raw = self._call_api([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        return self._parse_response(raw)

    def judge_raw(
        self,
        raw_response: str,
        ground_truth_mistake: str,
        ground_truth_feedback: str,
        instruction: str = "",
    ) -> JudgeResult:
        """Judge raw/malformed model response against ground truth.

        Use for samples where JSON parsing failed. The judge LLM classifies
        based on the intended content of the raw text.
        """
        user_prompt = JUDGE_RAW_USER_TEMPLATE.format(
            instruction=instruction,
            ground_truth_mistake=ground_truth_mistake,
            ground_truth_feedback=ground_truth_feedback or "",
            raw_response=raw_response,
        )
        raw = self._call_api([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        return self._parse_response(raw)

    def _call_api(self, messages: list[dict]) -> str:
        """Call the judge LLM with retry for transient errors.

        Permanent errors (auth, bad request, not found) propagate immediately.
        Transient errors (rate limit, timeout, connection, server) retry
        with exponential backoff.
        """
        client = self._get_client()

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0,
                    max_tokens=1000,
                )
                if not response.choices:
                    print(
                        "WARNING: API returned empty choices "
                        "(possible content filter or token exhaustion)",
                        file=sys.stderr,
                    )
                    return ""
                return response.choices[0].message.content or ""
            except PERMANENT_ERRORS:
                raise
            except _TRANSIENT_ERRORS as e:
                # insufficient_quota is 429 but not transient — fail fast
                if "insufficient_quota" in str(e):
                    raise
                if attempt == _MAX_RETRIES:
                    raise
                wait = _BACKOFF_BASE ** attempt
                print(
                    f"WARNING: Retry {attempt + 1}/{_MAX_RETRIES} "
                    f"after {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                time.sleep(wait)

        return ""  # unreachable, satisfies type checker

    def _parse_response(self, raw_response: str) -> JudgeResult:
        """Parse categorical JSON response from the judge LLM.

        Each dimension is parsed independently so one bad field does not
        poison the other.
        """
        text = raw_response.strip()

        # Strip markdown code blocks
        code_block = re.match(
            r"^```(?:\w*)\n(.*?)```\s*$", text, re.DOTALL,
        )
        if code_block:
            text = code_block.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: extract JSON object from free-text response
            json_match = re.search(
                r'\{[^{}]*"mistake"[^{}]*\}', text,
            )
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

            if data is None:
                if raw_response:
                    print(
                        f"WARNING: Judge returned unparseable JSON: "
                        f"{raw_response[:200]}",
                        file=sys.stderr,
                    )
                return JudgeResult(
                    mistake=None,
                    mistake_reasoning="JSON parse failed",
                    feedback=None,
                    feedback_reasoning="JSON parse failed",
                    raw_response=raw_response,
                    source=JudgeSource.LLM,
                )

        # Parse each dimension independently
        mistake = _extract_category(
            data, "mistake", QualityCategory,
        )
        feedback = _extract_category(
            data, "feedback", FeedbackCategory,
        )

        return JudgeResult(
            mistake=mistake[0],
            mistake_reasoning=mistake[1],
            feedback=feedback[0],
            feedback_reasoning=feedback[1],
            raw_response=raw_response,
            source=JudgeSource.LLM,
        )


def _extract_category(
    data: dict, field: str, enum_cls: type[StrEnum],
) -> tuple[StrEnum | None, str]:
    """Extract a categorical field from parsed JSON.

    Returns (category, reasoning). Category is None if invalid/missing.
    """
    raw_val = str(data.get(field, "")).strip().lower()
    reasoning = str(data.get(f"{field}_reasoning", ""))

    try:
        return enum_cls(raw_val), reasoning
    except ValueError:
        pass

    if raw_val:
        print(
            f"WARNING: Invalid {field} category: {raw_val!r}",
            file=sys.stderr,
        )
        reasoning = f"Invalid category '{raw_val}': {reasoning}"

    return None, reasoning
