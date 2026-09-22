import json
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from sklearn.model_selection import KFold


class Dataset(NamedTuple):
    compounds: list[dict[str, Any]]
    lc_condition: dict[str, Any]


def _dataset_path(dataset_id: str, data_dir: Path | str) -> Path:
    dataset_id = str(dataset_id)
    dataset_part = Path(dataset_id)
    if (
        not dataset_id
        or dataset_id in {".", ".."}
        or dataset_part.is_absolute()
        or dataset_part.name != dataset_id
    ):
        raise ValueError(f"dataset_id must be one directory name, got {dataset_id!r}")
    return Path(data_dir) / dataset_id


def load_dataset(
    dataset_id: str,
    data_dir: Path | str,
) -> Dataset:
    dataset_id = str(dataset_id)
    directory = _dataset_path(dataset_id, data_dir)
    path = directory / f"{dataset_id}_attributes.json"
    with path.open("r", encoding="utf-8") as handle:
        compounds = json.load(handle)
    if not isinstance(compounds, list) or not all(isinstance(row, dict) for row in compounds):
        raise ValueError(f"Expected a JSON list of compound objects in {path}")
    path = directory / f"{dataset_id}_lc_condition.yaml"
    with path.open("r", encoding="utf-8") as handle:
        condition = yaml.safe_load(handle)
    if not isinstance(condition, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return Dataset(compounds, condition)


def split_compounds(
    compounds: list[dict[str, Any]],
    *,
    fold_idx: int,
    seed: int,
    n_folds: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(fold_idx, int):
        raise ValueError(f"fold_idx must be an integer, got {fold_idx!r}")
    if not 0 <= fold_idx < n_folds:
        raise ValueError(f"fold_idx {fold_idx} out of range for n_folds={n_folds}")
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for index, (train_indices, test_indices) in enumerate(splitter.split(compounds)):
        if index == fold_idx:
            return (
                [compounds[int(row)] for row in train_indices],
                [compounds[int(row)] for row in test_indices],
            )
    raise RuntimeError("unreachable")
