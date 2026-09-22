from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import RTICLConfig, DEFAULT_CONFIG
from ..prompt import PromptPayload


@dataclass(frozen=True)
class ProviderResponse:
    """One provider response and its optional token accounting."""

    text: str | None
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    completed: bool = True


class Provider(Protocol):
    """Small async provider contract consumed by :mod:`rt_icl.pipeline`."""

    name: str
    model: str

    async def generate(self, payload: PromptPayload) -> ProviderResponse:
        pass

    def metadata(self) -> dict[str, Any]:
        pass


PROVIDER_NAMES = ("gemini", "openrouter", "openai")


def create_provider(
    provider_name: str,
    *,
    model: str | None = None,
    config: RTICLConfig = DEFAULT_CONFIG,
    timeout_seconds: float = 240.0,
    max_retries: int = 3,
) -> Provider:
    if provider_name == "gemini":
        from .gemini import GeminiProvider, GeminiSettings

        settings = GeminiSettings(
            model=GeminiSettings.model if model is None else model,
            max_output_tokens=config.max_output_tokens,
            thinking_level=config.thinking_level,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        return GeminiProvider(settings)

    if provider_name == "openrouter":
        from .openrouter import OPENROUTER_MODEL_PROFILES, OpenRouterProvider, OpenRouterSettings

        if model is None:
            available = ", ".join(sorted(OPENROUTER_MODEL_PROFILES))
            raise ValueError(
                f"model is required for OpenRouter; available: {available}"
            )
        settings = OpenRouterSettings.from_model(
            model,
            config=config,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        return OpenRouterProvider(settings)

    if provider_name == "openai":
        from .openai import OpenAIProvider, OpenAISettings

        settings = OpenAISettings(
            model=OpenAISettings.model if model is None else model,
            max_output_tokens=config.max_output_tokens,
            reasoning_effort=config.thinking_level,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        return OpenAIProvider(settings)

    available = ", ".join(PROVIDER_NAMES)
    raise ValueError(f"Unknown provider {provider_name!r}; available: {available}")


__all__ = ["Provider", "ProviderResponse", "PROVIDER_NAMES", "create_provider"]
