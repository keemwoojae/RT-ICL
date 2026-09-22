import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
import yaml

from .config import DEFAULT_DATASETS
from .data import _dataset_path
from .descriptors import compound_from_smiles
from .utils import as_finite_float


@dataclass(frozen=True)
class PreprocessingResult:
    compounds: list[dict[str, Any]]
    lc_condition: dict[str, Any]
    report: dict[str, Any]
    output_dir: Path


def _finite_rt(value: Any) -> float:
    number = as_finite_float(value)
    if number is None:
        raise ValueError("RT must be a finite number")
    return number


def _preprocess_rows(rows: Iterable[Mapping[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[tuple[dict[str, Any], str, float]]] = {}
    exclusions: list[dict[str, Any]] = []
    input_rows = 0
    for input_rows, row in enumerate(rows, 1):
        identity = {"row": input_rows, "id": row.get("id", "")}
        smiles = (row.get("pubchem.smiles.isomeric") or "").strip()
        if not smiles:
            smiles = (row.get("pubchem.smiles.canonical") or "").strip()
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            exclusions.append({**identity, "reason": "invalid_or_missing_smiles"})
            continue
        rt = as_finite_float(row.get("rt"))
        if rt is None:
            exclusions.append({**identity, "reason": "invalid_or_missing_rt"})
            continue
        formula = (row.get("formula") or "").strip()
        calculated = rdMolDescriptors.CalcMolFormula(mol)
        if not formula or re.sub(r"[+-]\d*$", "", formula) != re.sub(r"[+-]\d*$", "", calculated):
            exclusions.append({**identity, "reason": "missing_or_mismatched_formula"})
            continue
        canonical = Chem.MolToSmiles(mol)
        groups.setdefault(canonical, []).append((identity, row.get("name", ""), rt))

    compounds = []
    merged_rows = 0
    for smiles, group in groups.items():
        values = [entry[2] for entry in group]
        # Direct float comparison preserves the benchmark's historical boundary.
        if max(values) - min(values) > 0.1:
            exclusions.extend({**entry[0], "reason": "duplicate_rt_spread_gt_0.1_min"} for entry in group)
            continue
        record = compound_from_smiles(smiles, name=group[0][1])
        record["RT"] = round(round(statistics.mean(values), 6) * 60, 2)
        compounds.append(record)
        merged_rows += len(group) - 1
    return compounds, {
        "input_rows": input_rows,
        "output_compounds": len(compounds),
        "excluded_rows": len(exclusions),
        "merged_rows": merged_rows,
        "exclusions": sorted(exclusions, key=lambda entry: entry["row"]),
    }


def preprocess_dataset(
    dataset_id: str,
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> PreprocessingResult:
    source = _dataset_path(dataset_id, raw_dir)
    destination = _dataset_path(dataset_id, output_dir)
    table = source / f"{dataset_id}_rtdata.tsv"
    condition_path = source / f"{dataset_id}_lc_condition.yaml"
    filenames = (f"{dataset_id}_attributes.json", condition_path.name, "preprocessing_report.json")
    if source.resolve() == destination.resolve():
        raise ValueError("Raw and processed dataset directories must differ")
    if not overwrite and any((destination / name).exists() for name in filenames):
        raise FileExistsError(f"Processed files already exist in {destination}; use overwrite=True to replace them")
    condition_bytes = condition_path.read_bytes()
    condition = yaml.safe_load(condition_bytes)
    if not isinstance(condition, dict) or str(condition.get("column_type", "")).lower() not in {"reversed-phase", "hilic"}:
        raise ValueError(f"{condition_path} must specify column_type: Reversed-Phase or HILIC")
    with table.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or ())
        if not {"formula", "rt"} <= columns or not columns.intersection({"pubchem.smiles.isomeric", "pubchem.smiles.canonical"}):
            raise ValueError(f"{table} is missing required RepoRT columns")
        compounds, report = _preprocess_rows(reader)
    if not compounds:
        raise ValueError(f"No eligible compounds remain in {table}")
    report.update({
        "dataset_id": dataset_id,
        "rdkit_version": rdBase.rdkitVersion,
        "raw_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "lc_condition_sha256": hashlib.sha256(condition_bytes).hexdigest(),
        "rt_unit": "seconds",
    })
    destination.mkdir(parents=True, exist_ok=True)
    (destination / filenames[0]).write_text(json.dumps(compounds, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (destination / filenames[1]).write_bytes(condition_bytes)
    (destination / filenames[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return PreprocessingResult(compounds, condition, report, destination)


def load_compound_table(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("Compound tables must use .csv or .tsv")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))


def prepare_compounds(
    rows: Iterable[Mapping[str, Any]],
    *,
    rt_unit: str,
    require_rt: bool = False,
) -> list[dict[str, Any]]:
    if rt_unit not in {"seconds", "minutes"}:
        raise ValueError("rt_unit must be 'seconds' or 'minutes'")
    compounds = []
    for index, row in enumerate(rows, 1):
        smiles = row.get("SMILES", row.get("smiles"))
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"Row {index}: missing SMILES")
        try:
            record = compound_from_smiles(smiles.strip(), name=row.get("Name", row.get("name")))
            rt = row.get("RT", row.get("rt"))
            if rt is None or (isinstance(rt, str) and not rt.strip()):
                if require_rt:
                    raise ValueError("reference RT is required")
            else:
                value = _finite_rt(rt)
                value *= 60 if rt_unit == "minutes" else 1
                record["RT"] = _finite_rt(value)
        except ValueError as exc:
            raise ValueError(f"Row {index}: {exc}") from exc
        compounds.append(record)
    return compounds


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare bundled RepoRT datasets.")
    parser.add_argument("datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--raw-dir", type=Path, default=root / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for dataset_id in args.datasets:
        result = preprocess_dataset(dataset_id, args.raw_dir, args.output_dir, overwrite=args.overwrite)
        report = result.report
        print(f"{dataset_id}: {report['input_rows']} rows -> {report['output_compounds']} compounds; "
              f"{report['excluded_rows']} excluded, {report['merged_rows']} merged")


if __name__ == "__main__":
    main()
