from typing import Optional, Type

from music_evalkit.models.base import BaseLLMClient, LLMConfig
from music_evalkit.models.config import AppConfig


# Mapping from provider name to client class name
# Multiple providers can use the same underlying client
PROVIDER_CLIENT_MAP = {
    # OpenAI-compatible providers
    "openai": "openai_compat",
    "openai_audio": "openai_compat",
    "gemini": "openai_compat",
    "qwen": "openai_compat",
    "ollama": "openai_compat",
    # NVIDIA Flamingo models (local)
    "audio_flamingo": "audio_flamingo",
    "music_flamingo": "music_flamingo",
    # Qwen2.5-Omni (local)
    "qwen_local": "qwen_local",
    # Testing
    "noop": "noop",
}

# Global registry of client classes
_registry: dict[str, Type[BaseLLMClient]] = {}


def register_client(name: str):
    """Decorator to register an LLM client class.

    Args:
        name: Client name to register under (e.g., 'openai_compat', 'huggingface').
    """
    def decorator(cls: Type[BaseLLMClient]) -> Type[BaseLLMClient]:
        if name in _registry:
            raise ValueError(f"Client '{name}' is already registered")
        _registry[name] = cls
        return cls
    return decorator


def get_client(
    provider: str,
    app_config: AppConfig,
    model: Optional[str] = None,
) -> BaseLLMClient:
    """Factory to get LLM client by provider name.

    Args:
        provider: Provider name from config (openai, gemini, qwen, huggingface, etc.)
        app_config: Application configuration with provider settings.
        model: Optional model override. If not specified, uses default from config.

    Returns:
        Configured LLM client instance.

    Usage:
        app_config = AppConfig.load()
        client = get_client("gemini", app_config)
        client = get_client("huggingface", app_config, model="Qwen/Qwen2-Audio-7B")
    """
    # Get config for this provider
    llm_config = app_config.providers.get(provider)
    if llm_config is None:
        available = list(app_config.providers.keys())
        raise ValueError(f"Provider '{provider}' not found in config. Available: {available}")

    # Map provider name to client class name
    client_name = PROVIDER_CLIENT_MAP.get(provider)
    if client_name is None:
        raise ValueError(
            f"Provider '{provider}' has no client mapping. "
            f"Add it to PROVIDER_CLIENT_MAP in factory.py"
        )

    # Ensure clients are imported
    if client_name not in _registry:
        _import_providers()

    if client_name not in _registry:
        registered = list(_registry.keys())
        raise ValueError(f"Client '{client_name}' not registered. Registered: {registered}")

    # Override model if specified
    if model:
        llm_config = llm_config.model_copy(update={"model_name": model})

    return _registry[client_name](llm_config)


def _import_providers() -> None:
    """Import all provider modules to trigger registration."""
    from music_evalkit.models.providers import openai_compat  # noqa: F401
    from music_evalkit.models.providers import flamingo  # noqa: F401
    from music_evalkit.models.providers import noop  # noqa: F401
    from music_evalkit.models.providers import qwen_local  # noqa: F401


def list_providers(app_config: AppConfig) -> list[str]:
    """List available providers from config."""
    return list(app_config.providers.keys())


def list_registered_clients() -> list[str]:
    """List registered client implementations."""
    _import_providers()
    return list(_registry.keys())
