import asyncio
from dataclasses import asdict, dataclass
import os
from typing import Any, Literal

from ..config import RTICLConfig, DEFAULT_CONFIG
from ..prompt import PromptPayload
from . import ProviderResponse


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
ReasoningEffort = Literal["low"]


@dataclass(frozen=True)
class OpenRouterModelProfile:
    model: str
    serving_provider: str
    reasoning_effort: ReasoningEffort | None


OPENROUTER_MODEL_PROFILES: dict[str, OpenRouterModelProfile] = {
    "qwen/qwen3-235b-a22b-2507": OpenRouterModelProfile(
        model="qwen/qwen3-235b-a22b-2507",
        serving_provider="Alibaba",
        reasoning_effort=None,
    ),
    "openai/gpt-oss-120b": OpenRouterModelProfile(
        model="openai/gpt-oss-120b",
        serving_provider="Groq",
        reasoning_effort="low",
    ),
}


@dataclass(frozen=True)
class OpenRouterSettings:
    """Reproducible generation and routing settings for one OpenRouter model."""

    model: str
    serving_provider: str
    max_output_tokens: int = 8192
    reasoning_effort: ReasoningEffort | None = None
    allow_fallbacks: bool = False
    timeout_seconds: float = 240.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        profile = OPENROUTER_MODEL_PROFILES.get(self.model)
        if profile is None:
            available = ", ".join(sorted(OPENROUTER_MODEL_PROFILES))
            raise ValueError(
                f"Unsupported OpenRouter model {self.model!r}; available: {available}"
            )
        if self.serving_provider != profile.serving_provider:
            raise ValueError(
                f"{self.model} is pinned to {profile.serving_provider!r}, "
                f"got {self.serving_provider!r}"
            )
        if self.reasoning_effort != profile.reasoning_effort:
            raise ValueError(
                f"{self.model} requires reasoning_effort="
                f"{profile.reasoning_effort!r}"
            )
        if self.max_output_tokens != 8192:
            raise ValueError(
                "OpenRouter comparison models are fixed to max_output_tokens=8192"
            )
        if self.allow_fallbacks:
            raise ValueError("OpenRouter comparison models disable provider fallbacks")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")

    @classmethod
    def from_model(
        cls,
        model: str,
        *,
        config: RTICLConfig = DEFAULT_CONFIG,
        timeout_seconds: float = 240.0,
        max_retries: int = 3,
    ) -> "OpenRouterSettings":
        try:
            profile = OPENROUTER_MODEL_PROFILES[model]
        except KeyError as exc:
            available = ", ".join(sorted(OPENROUTER_MODEL_PROFILES))
            raise ValueError(
                f"Unsupported OpenRouter model {model!r}; available: {available}"
            ) from exc
        if profile.reasoning_effort is None and config.thinking_level not in (None, "low"):
            raise ValueError(f"{model} does not support thinking_level={config.thinking_level!r}")
        return cls(
            model=profile.model,
            serving_provider=profile.serving_provider,
            max_output_tokens=config.max_output_tokens,
            reasoning_effort=config.thinking_level if profile.reasoning_effort is not None else None,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


class OpenRouterProvider:
    """Async OpenRouter provider with model and serving-provider pins."""

    name = "openrouter"

    def __init__(
        self,
        settings: OpenRouterSettings,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.model = settings.model
        self._api_key = api_key
        self._client = client

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.name, **asdict(self.settings),
            "reasoning_enabled": self.settings.reasoning_effort is not None,
            "require_parameters": True,
        }

    def ensure_available(self) -> None:
        self._get_client()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import openai

        api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._client = openai.AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            max_retries=0,
        )
        return self._client

    async def generate(self, payload: PromptPayload) -> ProviderResponse:
        client = self._get_client()
        extra_body: dict[str, Any] = {
            "provider": {
                "order": [self.settings.serving_provider],
                "allow_fallbacks": self.settings.allow_fallbacks,
                "require_parameters": True,
            },
        }
        if self.settings.reasoning_effort is not None:
            extra_body["reasoning"] = {"effort": self.settings.reasoning_effort}

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": payload.user_content}],
            "max_tokens": self.settings.max_output_tokens,
            "extra_body": extra_body,
        }

        for attempt in range(self.settings.max_retries):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=self.settings.timeout_seconds,
                )
                choices = getattr(response, "choices", None)
                choice = choices[0] if choices else None
                text = choice.message.content if choice is not None else None
                finish_reason = getattr(choice, "finish_reason", None)
                completed = finish_reason == "stop"
                read_error = None if text is not None else "unreadable_response: no content"
                return ProviderResponse(
                    text=str(text) if text is not None else None,
                    usage=_usage(getattr(response, "usage", None)),
                    error=(
                        read_error if completed else
                        f"incomplete_response: finish_reason={finish_reason!r}"
                    ),
                    metadata={
                        "response_id": getattr(response, "id", None),
                        "response_model": getattr(response, "model", None),
                        "serving_provider": getattr(response, "provider", None),
                        "finish_reason": finish_reason,
                    },
                    completed=completed,
                )
            except Exception as exc:
                if attempt >= self.settings.max_retries - 1 or not _is_retryable(exc):
                    raise
                await asyncio.sleep(10)


def _usage(usage: Any | None) -> dict[str, int]:
    if usage is None:
        return {}
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = (
        getattr(details, "reasoning_tokens", 0) or 0
        if details is not None
        else 0
    )
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "thinking_tokens": int(reasoning_tokens),
    }


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    return (
        getattr(exc, "status_code", None) in RETRYABLE_HTTP_CODES
        or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
    )


__all__ = [
    "OPENROUTER_MODEL_PROFILES",
    "OpenRouterModelProfile",
    "OpenRouterProvider",
    "OpenRouterSettings",
]
