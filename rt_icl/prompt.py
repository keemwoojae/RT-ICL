from dataclasses import dataclass, field
import hashlib
from importlib import resources
import math
from typing import Any, Sequence


COMPOUND_KEYS = (
    "SMILES",
    "Name",
    "Mol_Formula",
    "Exact_Mass",
    "LogP",
    "TPSA",
    "HBA_Count",
    "HBD_Count",
    "Aromatic_Ring_Count",
)


_COMPOUND_LABELS = {
    "Mol_Formula": "Molecular formula",
    "Exact_Mass": "Exact mass",
    "HBA_Count": "HBA count",
    "HBD_Count": "HBD count",
    "Aromatic_Ring_Count": "Aromatic ring count",
}


def format_lc_condition(lc_condition: dict[str, Any]) -> str:
    scalar_keys = [
        ("column_type", "LC mode"),
        ("column", "Column"),
        ("column_temperature", "Column temperature"),
    ]
    scalar_keys.extend(
        (key, f"Mobile phase {key.removeprefix('mobile_phase_')}")
        for key in sorted(key for key in lc_condition if key.startswith("mobile_phase_"))
    )
    scalar_keys.append(("flow_rate", "Flow rate"))

    lines: list[str] = []
    for key, label in scalar_keys:
        value = lc_condition.get(key, "")
        if value is None or value == "":
            continue
        value_text = str(value)
        if key == "column_temperature" and value_text.strip().lower() in {
            "na",
            "n/a",
            "none",
            "null",
        }:
            continue
        lines.append(f"{label}: {value_text}")

    gradient = lc_condition.get("gradient", [])
    if gradient:
        lines.append("Gradient:")
        lines.extend(f"  {step}" for step in gradient)
    return "\n".join(lines)


def format_compound(compound: dict[str, Any], *, include_rt: bool = False) -> str:
    keys = [*COMPOUND_KEYS, "RT"] if include_rt else list(COMPOUND_KEYS)
    lines = []
    for key in keys:
        value = compound.get(key)
        if value is None:
            continue
        lines.append(f"{_COMPOUND_LABELS.get(key, key)}: {value}")
    return "\n".join(lines)


def _format_examples(examples: Sequence[dict[str, Any]]) -> str:
    intro = "The following reference compounds were retrieved from the reference dataset as the most structurally similar to the query compound based on Morgan-fingerprint Tanimoto similarity."
    blocks = []
    for index, example in enumerate(examples, 1):
        marker = f"[Reference compound {index}]"
        if "Tanimoto" in example:
            marker = f"[Reference compound {index} | Tanimoto similarity {float(example['Tanimoto']):.2f}]"
        blocks.append(f"{marker}\n{format_compound(example, include_rt=True)}")
    return intro + "\n\n" + "\n\n".join(blocks)


@dataclass(frozen=True)
class PromptPayload:
    """The exact user message and its UTF-8 SHA-256 digest."""

    user_content: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sha256", hashlib.sha256(self.user_content.encode("utf-8")).hexdigest()
        )


def _read_prompt_asset(name: str) -> str:
    return (
        resources.files("rt_icl").joinpath("prompts", name)
        .read_text(encoding="utf-8").rstrip("\r\n")
    )


def build_prompt(
    query: dict[str, Any],
    examples: Sequence[dict[str, Any]],
    lc_condition: dict[str, Any],
    *,
    reference_rt_range: tuple[float, float],
) -> PromptPayload:
    if not examples:
        raise ValueError("RT-ICL requires selected demonstrations")
    try:
        rt_min, rt_max = (float(value) for value in reference_rt_range)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"reference_rt_range must be a (min, max) pair of numbers, got {reference_rt_range!r}"
        ) from exc
    if not (math.isfinite(rt_min) and math.isfinite(rt_max)):
        raise ValueError(f"reference_rt_range must be finite, got {reference_rt_range!r}")
    if rt_min > rt_max:
        raise ValueError(f"reference_rt_range min exceeds max: {reference_rt_range!r}")
    rt_min, rt_max = f"{rt_min:.1f}", f"{rt_max:.1f}"

    mode = str(lc_condition.get("column_type", "")).lower()
    guidance_asset = {
        "reversed-phase": "rplc_instruction.txt",
        "hilic": "hilic_instruction.txt",
    }.get(mode)
    if guidance_asset is None:
        raise ValueError(
            "RT-ICL requires column_type to be either 'Reversed-Phase' or 'HILIC', "
            f"got {lc_condition.get('column_type')!r}"
        )

    def fill(text: str) -> str:
        return text.replace("{rt_min}", rt_min).replace("{rt_max}", rt_max)

    sections = [
        _read_prompt_asset("role.txt"),
        f"<Conditions>\n{format_lc_condition(lc_condition)}\n</Conditions>",
        fill(_read_prompt_asset(guidance_asset))
        + "\n" + fill(_read_prompt_asset("output.txt")),
        f"<Query>\n{format_compound(query)}\n</Query>",
        f"<Examples>\n{_format_examples(examples)}\n</Examples>",
    ]
    return PromptPayload("\n\n".join(sections))
