import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .utils import as_finite_float


RESULT_FILENAMES = ("run_metadata.json", "metrics.json", "predictions.csv")


def check_result_conflicts(output_dir: str | Path) -> Path:
    run_path = Path(output_dir)
    for path in (run_path, *run_path.parents):
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"Result output path is not a directory: {path}")
    conflicts = [
        run_path / name for name in RESULT_FILENAMES if (run_path / name).exists()
    ]
    if conflicts:
        joined = ", ".join(path.name for path in conflicts)
        raise FileExistsError(
            f"Refusing to overwrite existing result artifacts in {run_path}: {joined}. "
            "Choose another output directory."
        )
    return run_path


def compute_metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    parsed = sum(as_finite_float(row.get("pred_rt")) is not None for row in predictions)
    api_errors = sum(row.get("parse_status") == "api_error" for row in predictions)
    labeled = sum(as_finite_float(row.get("actual_rt")) is not None for row in predictions)
    pairs = [
        (actual, predicted)
        for row in predictions
        if (actual := as_finite_float(row.get("actual_rt"))) is not None
        and (predicted := as_finite_float(row.get("pred_rt"))) is not None
    ]

    metrics: dict[str, Any] = {
        "total": total,
        "parsed": parsed,
        "parse_failures": total - parsed,
        "parse_rate": parsed / total if total else None,
        "api_errors": api_errors,
        "labeled": labeled,
        "evaluated": len(pairs),
        "mae": None,
        "median_absolute_error": None,
        "mape_percent": None,
        "r2": None,
    }
    if not pairs:
        return metrics

    actuals = [pair[0] for pair in pairs]
    predicted = [pair[1] for pair in pairs]
    absolute_errors = [abs(pred - actual) for actual, pred in pairs]
    metrics["mae"] = sum(absolute_errors) / len(absolute_errors)
    metrics["median_absolute_error"] = statistics.median(absolute_errors)

    nonzero = [
        abs(pred - actual) / abs(actual)
        for actual, pred in pairs
        if actual != 0
    ]
    metrics["mape_percent"] = (
        100.0 * sum(nonzero) / len(nonzero) if nonzero else None
    )

    mean_actual = sum(actuals) / len(actuals)
    denominator = sum((actual - mean_actual) ** 2 for actual in actuals)
    if denominator > 0:
        numerator = sum(
            (actual - pred) ** 2 for actual, pred in zip(actuals, predicted)
        )
        metrics["r2"] = 1.0 - numerator / denominator
    return metrics


def build_metrics(
    predictions: Sequence[Mapping[str, Any]],
    *,
    include_folds: bool = False,
    token_usage: Mapping[str, int] | None = None,
    timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {"overall": compute_metrics(predictions)}
    if include_folds:
        fold_predictions: dict[int, list[Mapping[str, Any]]] = {}
        for row in predictions:
            fold = row.get("fold")
            if fold is not None and fold != "":
                fold_predictions.setdefault(int(fold), []).append(row)
        output["folds"] = [
            {"fold": fold, **compute_metrics(rows)}
            for fold, rows in sorted(fold_predictions.items())
        ]
    if token_usage:
        output["token_usage"] = dict(token_usage)
    if timing:
        output["timing"] = dict(timing)
    return output


def save_results(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> Path:
    run_path = check_result_conflicts(output_dir)

    run_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(run_path / "run_metadata.json", config)
    _atomic_json(run_path / "metrics.json", metrics)

    preferred = [
        "fold",
        "index",
        "source_index",
        "SMILES",
        "actual_rt",
        "pred_rt",
        "abs_error",
        "parse_status",
        "response",
        "error",
        "prompt_sha256",
        "examples",
        "usage",
        "response_metadata",
    ]
    keys = {key for row in predictions for key in row}
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    # Preserve a useful header even for an empty inference input.
    if not fieldnames:
        fieldnames = preferred

    path = run_path / "predictions.csv"
    temp = path.with_name(f".{path.name}.tmp")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in predictions:
                values = {key: row.get(key) for key in fieldnames}
                for key, value in values.items():
                    if isinstance(value, (dict, list, tuple)):
                        values[key] = json.dumps(
                            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                        )
                writer.writerow(values)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return run_path


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


__all__ = [
    "build_metrics",
    "check_result_conflicts",
    "compute_metrics",
    "save_results",
]
