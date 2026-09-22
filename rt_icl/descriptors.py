from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def compound_from_smiles(
    smiles: str,
    *,
    name: str | None = None,
    decimals: int = 2,
) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(canonical)

    compound: dict[str, Any] = {
        "SMILES": canonical,
        "Mol_Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Exact_Mass": round(Descriptors.ExactMolWt(mol), decimals),
        "LogP": round(Descriptors.MolLogP(mol), decimals),
        "TPSA": round(Descriptors.TPSA(mol), decimals),
        "HBA_Count": Lipinski.NumHAcceptors(mol),
        "HBD_Count": Lipinski.NumHDonors(mol),
        "Aromatic_Ring_Count": rdMolDescriptors.CalcNumAromaticRings(mol),
    }
    if name is not None:
        compound["Name"] = name
    return compound
