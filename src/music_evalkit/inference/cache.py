import hashlib
import json
from pathlib import Path
from typing import Optional

from music_evalkit.models.base import LLMResponse


class InferenceCache:
    """JSONL-backed cache with atomic writes.

    Each model gets its own cache file for easy management.
    Cache keys are deterministic hashes of model + prompt + audio paths.
    """

    def __init__(self, cache_dir: Path = Path("data/cache")):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[str, LLMResponse]] = {}

    def _get_cache_file(self, model: str) -> Path:
        """Get cache file path for a model.

        Args:
            model: Model name.

        Returns:
            Path to cache file.
        """
        # Sanitize model name for filename
        safe_name = model.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_name}_cache.jsonl"

    def make_key(self, model: str, prompt: str, audio_paths: list[Path]) -> str:
        """Generate cache key from model + prompt + audio hashes.

        Args:
            model: Model name.
            prompt: Prompt text.
            audio_paths: List of audio file paths.

        Returns:
            Deterministic cache key.
        """
        # Hash audio files by content
        audio_hashes = []
        for path in sorted(audio_paths):
            if path.exists():
                with open(path, "rb") as f:
                    audio_hashes.append(hashlib.md5(f.read()).hexdigest())
            else:
                audio_hashes.append(str(path))

        # Combine all components
        key_data = {
            "model": model,
            "prompt": prompt,
            "audio_hashes": audio_hashes,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def _load_cache_file(self, model: str) -> dict[str, LLMResponse]:
        """Load cache file into memory.

        Args:
            model: Model name.

        Returns:
            Dictionary mapping cache keys to responses.
        """
        if model in self._memory_cache:
            return self._memory_cache[model]

        cache = {}
        cache_file = self._get_cache_file(model)

        if cache_file.exists():
            with open(cache_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        key = data["key"]
                        response = LLMResponse(**data["response"])
                        cache[key] = response
                    except (json.JSONDecodeError, KeyError):
                        continue

        self._memory_cache[model] = cache
        return cache

    def get(self, key: str, model: str) -> Optional[LLMResponse]:
        """Retrieve cached response.

        Args:
            key: Cache key.
            model: Model name.

        Returns:
            Cached LLMResponse or None if not found.
        """
        cache = self._load_cache_file(model)
        return cache.get(key)

    def set(self, key: str, model: str, response: LLMResponse) -> None:
        """Store response in cache.

        Uses atomic write (write to temp, then append) for safety.

        Args:
            key: Cache key.
            model: Model name.
            response: Response to cache.
        """
        # Update memory cache
        if model not in self._memory_cache:
            self._load_cache_file(model)
        self._memory_cache[model][key] = response

        # Append to file
        cache_file = self._get_cache_file(model)
        entry = {
            "key": key,
            "response": response.model_dump(),
        }

        with open(cache_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_all(self, model: str) -> dict[str, LLMResponse]:
        """Load entire cache into memory for faster lookups.

        Args:
            model: Model name.

        Returns:
            Dictionary mapping cache keys to responses.
        """
        return self._load_cache_file(model)

    def clear(self, model: Optional[str] = None) -> None:
        """Clear cache.

        Args:
            model: Model name to clear, or None to clear all.
        """
        if model:
            cache_file = self._get_cache_file(model)
            if cache_file.exists():
                cache_file.unlink()
            if model in self._memory_cache:
                del self._memory_cache[model]
        else:
            for cache_file in self.cache_dir.glob("*_cache.jsonl"):
                cache_file.unlink()
            self._memory_cache.clear()

    def stats(self, model: str) -> dict:
        """Get cache statistics for a model.

        Args:
            model: Model name.

        Returns:
            Dictionary with cache statistics.
        """
        cache = self._load_cache_file(model)
        cache_file = self._get_cache_file(model)

        return {
            "entries": len(cache),
            "file_size_bytes": cache_file.stat().st_size if cache_file.exists() else 0,
            "file_path": str(cache_file),
        }
