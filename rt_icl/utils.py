from dataclasses import dataclass
import math
import re
from typing import Any, Literal


def as_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


_RT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ParseResult:
    value: float | None
    status: Literal["ok", "empty", "failed"]


def parse_rt(response: str | None) -> ParseResult:
    if response is None or response == "":
        return ParseResult(None, "empty")
    match = _RT_PATTERN.fullmatch(response.strip())
    value = as_finite_float(match.group(0)) if match else None
    return ParseResult(value, "ok" if value is not None else "failed")
