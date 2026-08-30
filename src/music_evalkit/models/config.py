import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from music_evalkit.models.base import LLMConfig


def expand_env_vars(value: str) -> str:
    """Expand ${VAR_NAME} patterns in string."""
    pattern = r"\$\{([^}]+)\}"

    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return re.sub(pattern, replace, value)


def expand_recursive(obj: Any) -> Any:
    """Recursively expand env vars in all string values."""
    if isinstance(obj, str):
        return expand_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: expand_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_recursive(item) for item in obj]
    return obj


class InferenceConfig(BaseModel):
    """Configuration for inference runs."""

    max_concurrent: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_dir: str = "data/cache"


class AppConfig(BaseModel):
    """Full application config."""

    providers: dict[str, LLMConfig]
    inference: InferenceConfig

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppConfig":
        """Load config from YAML file with env var expansion.

        Args:
            config_path: Path to config.yaml. If None, searches for it.

        Returns:
            AppConfig instance.
        """
        load_dotenv()

        if config_path is None:
            config_path = find_config_file()

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        config = expand_recursive(raw_config)

        # Parse providers - convert default_model to model_name
        providers = {}
        for name, data in config.get("providers", {}).items():
            # Map default_model -> model_name for LLMConfig
            if "default_model" in data:
                data["model_name"] = data.pop("default_model")
            providers[name] = LLMConfig(**data)

        # Parse inference config
        inference = InferenceConfig(**config.get("inference", {}))

        return cls(providers=providers, inference=inference)


def find_config_file() -> Path:
    """Find config.yaml by searching up from current directory."""
    current = Path.cwd()
    while current != current.parent:
        config_path = current / "config.yaml"
        if config_path.exists():
            return config_path
        current = current.parent

    # Default to current directory
    return Path("config.yaml")