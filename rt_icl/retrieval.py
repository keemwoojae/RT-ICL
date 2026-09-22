from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .config import DEFAULT_CONFIG


SIMILARITY_KEY = "Tanimoto"


@lru_cache(maxsize=16)
def _morgan_generator(fp_radius: int, fp_nbits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_nbits)


@lru_cache(maxsize=16384)
def compute_fingerprint(
    smiles: str,
    fp_radius: int = DEFAULT_CONFIG.fp_radius,
    fp_nbits: int = DEFAULT_CONFIG.fp_nbits,
):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return None if mol is None else _morgan_generator(fp_radius, fp_nbits).GetFingerprint(mol)


def select_tanimoto_examples(
    query: dict[str, Any],
    references: Sequence[dict[str, Any]],
    k: int = DEFAULT_CONFIG.n_shot,
    *,
    fp_radius: int = DEFAULT_CONFIG.fp_radius,
    fp_nbits: int = DEFAULT_CONFIG.fp_nbits,
) -> list[dict[str, Any]]:
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    pool = list(references)
    if k == 0 or not pool:
        return []

    query_smiles = query["SMILES"]
    query_fp = compute_fingerprint(str(query_smiles), fp_radius, fp_nbits)
    if query_fp is None:
        raise ValueError(f"Invalid query SMILES: {query_smiles!r}")

    similarities = np.empty(len(pool))
    for index, reference in enumerate(pool):
        reference_fp = compute_fingerprint(str(reference.get("SMILES", "")), fp_radius, fp_nbits)
        similarities[index] = (
            0.0
            if reference_fp is None
            else float(DataStructs.TanimotoSimilarity(query_fp, reference_fp))
        )

    # This small-pool branch is intentionally RT-sorted: it is the historical behavior.
    if len(pool) <= k:
        selected_indices = sorted(range(len(pool)), key=lambda index: float(pool[index]["RT"]))
    else:
        # Keep NumPy's original argsort/reverse expression so ties follow the frozen implementation.
        selected_indices = np.argsort(similarities)[::-1][:k]
    return [
        {**pool[index], SIMILARITY_KEY: round(float(similarities[index]), 4)}
        for index in selected_indices
    ]
