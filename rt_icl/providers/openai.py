import asyncio
from dataclasses import asdict, dataclass
import os
from typing import Any, Literal

from ..prompt import PromptPayload
from . import ProviderResponse


OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
ReasoningEffort = Literal["low"]


@dataclass(frozen=True)
class OpenAISettings:
    """Fixed settings for the paper's direct OpenAI comparison model."""

    model: str = OPENAI_MODEL
    max_output_tokens: int = 8192
    reasoning_effort: ReasoningEffort = "low"
    store: bool = False
    timeout_seconds: float = 240.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if self.model != OPENAI_MODEL:
            raise ValueError(
                f"Direct OpenAI comparison is fixed to {OPENAI_MODEL!r}, "
                f"got {self.model!r}"
            )
        if self.max_output_tokens != 8192:
            raise ValueError(
                "Direct OpenAI comparison is fixed to max_output_tokens=8192"
            )
        if self.reasoning_effort != "low":
            raise ValueError("Direct OpenAI comparison is fixed to reasoning_effort='low'")
        if self.store:
            raise ValueError("Direct OpenAI comparison is fixed to store=False")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")


class OpenAIProvider:
    """Async provider using the direct OpenAI Responses API."""

    name = "openai"

    def __init__(
        self,
        settings: OpenAISettings | None = None,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or OpenAISettings()
        self.model = self.settings.model
        self._api_key = api_key
        self._client = client

    def metadata(self) -> dict[str, Any]:
        return {"provider": self.name, "api": "responses", **asdict(self.settings)}

    def ensure_available(self) -> None:
        self._get_client()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import openai

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
        return self._client

    async def generate(self, payload: PromptPayload) -> ProviderResponse:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": payload.user_content,
            "reasoning": {"effort": self.settings.reasoning_effort},
            "max_output_tokens": self.settings.max_output_tokens,
            "store": self.settings.store,
        }

        for attempt in range(self.settings.max_retries):
            try:
                response = await asyncio.wait_for(
                    client.responses.create(**kwargs),
                    timeout=self.settings.timeout_seconds,
                )
                text = getattr(response, "output_text", None)
                status = getattr(response, "status", None)
                incomplete_reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
                api_error = getattr(response, "error", None)
                error_message = getattr(api_error, "message", None)
                completed = status == "completed"
                read_error = None if text is not None else "unreadable_response: no output_text"
                return ProviderResponse(
                    text=str(text) if text is not None else None,
                    usage=_usage(getattr(response, "usage", None)),
                    error=(
                        read_error if completed else
                        f"incomplete_response: status={status!r}, reason={incomplete_reason or error_message!r}"
                    ),
                    metadata={
                        "response_id": getattr(response, "id", None),
                        "response_model": getattr(response, "model", None),
                        "service_tier": getattr(response, "service_tier", None),
                        "status": status,
                        "incomplete_reason": incomplete_reason,
                        "error_code": getattr(api_error, "code", None),
                        "error_message": error_message,
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
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = (
        getattr(details, "reasoning_tokens", 0) or 0
        if details is not None
        else 0
    )
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "thinking_tokens": int(reasoning_tokens),
    }


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    return (
        getattr(exc, "status_code", None) in RETRYABLE_HTTP_CODES
        or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
    )


__all__ = ["OPENAI_MODEL", "OpenAIProvider", "OpenAISettings"]
