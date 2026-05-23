from __future__ import annotations

import hashlib

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def smiles_to_fingerprint(smiles: str, radius: int, length: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid smiles: {smiles}")
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=length)
    arr = np.zeros((length,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def reaction_hash(reactants_smiles: str, products_smiles: str) -> str:
    reactants = Chem.MolFromSmiles(reactants_smiles)
    products = Chem.MolFromSmiles(products_smiles)
    if reactants is None or products is None:
        return hashlib.sha224(f"{reactants_smiles}++{products_smiles}".encode()).hexdigest()
    payload = Chem.MolToInchi(reactants) + "++" + Chem.MolToInchi(products)
    return hashlib.sha224(payload.encode("utf8")).hexdigest()

