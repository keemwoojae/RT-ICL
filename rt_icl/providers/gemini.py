import asyncio
import os
from dataclasses import asdict, dataclass
from typing import Any

from ..prompt import PromptPayload
from . import ProviderResponse


RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class GeminiSettings:
    """Generation settings for Gemini."""

    model: str = "gemini-3-flash-preview"
    max_output_tokens: int = 8192
    thinking_level: str | None = "low"
    timeout_seconds: float = 240.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")


class GeminiProvider:
    """Async Gemini provider accepting a single :class:`PromptPayload`."""

    name = "gemini"

    def __init__(
        self,
        settings: GeminiSettings | None = None,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or GeminiSettings()
        self.model = self.settings.model
        self._api_key = api_key
        self._client = client

    def metadata(self) -> dict[str, Any]:
        return {"provider": self.name, **asdict(self.settings)}

    def ensure_available(self) -> None:
        self._get_client()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY to use Gemini.")

        from google import genai

        self._client = genai.Client(api_key=api_key)
        return self._client

    async def generate(self, payload: PromptPayload) -> ProviderResponse:
        client = self._get_client()
        # The RT-ICL output contract is a bare number, so JSON response MIME types
        # and response schemas are deliberately absent.
        config: dict[str, Any] = {
            "max_output_tokens": self.settings.max_output_tokens,
        }
        if self.settings.thinking_level:
            config["thinking_config"] = {
                "thinking_level": self.settings.thinking_level,
            }

        for attempt in range(self.settings.max_retries):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=self.model,
                        contents=payload.user_content,
                        config=config,
                    ),
                    timeout=self.settings.timeout_seconds,
                )
                text, read_error = _response_text(response)
                candidates = getattr(response, "candidates", None)
                finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
                block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
                completed = finish_reason == "STOP"
                return ProviderResponse(
                    text=text,
                    usage=_usage(getattr(response, "usage_metadata", None)),
                    error=(
                        read_error if completed else
                        f"incomplete_response: finish_reason={finish_reason!r}, block_reason={block_reason!r}"
                    ),
                    metadata={
                        "response_id": getattr(response, "response_id", None),
                        "response_model": getattr(response, "model_version", None),
                        "finish_reason": finish_reason,
                        "block_reason": block_reason,
                    },
                    completed=completed,
                )
            except Exception as exc:
                if attempt >= self.settings.max_retries - 1 or not _is_retryable(exc):
                    raise
                await asyncio.sleep(10)


def _response_text(response: Any) -> tuple[str | None, str | None]:
    try:
        text = response.text
    except (AttributeError, IndexError, ValueError) as exc:
        return None, f"unreadable_response: {type(exc).__name__}: {exc}"
    if text is None:
        return None, "unreadable_response: response.text is None"
    return str(text), None


def _usage(metadata: Any | None) -> dict[str, int]:
    if metadata is None:
        return {}
    thinking_tokens = int(getattr(metadata, "thoughts_token_count", 0) or 0)
    return {
        "input_tokens": int(getattr(metadata, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(metadata, "candidates_token_count", 0) or 0) + thinking_tokens,
        "thinking_tokens": thinking_tokens,
    }


def _is_retryable(exc: Exception) -> bool:
    from httpx import NetworkError, RemoteProtocolError, TimeoutException

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, NetworkError, RemoteProtocolError, TimeoutException)):
        return True
    code = getattr(exc, "code", None)
    if code in RETRYABLE_HTTP_CODES:
        return True
    # Avoid importing google.genai.errors solely for an isinstance check.
    return type(exc).__name__ in {"ServerError", "ServiceUnavailable"}


__all__ = ["GeminiProvider", "GeminiSettings"]
