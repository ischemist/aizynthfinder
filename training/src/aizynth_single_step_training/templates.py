from __future__ import annotations

from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pandas as pd
from rxnutils.chem.reaction import ChemicalReaction, ReactionException
from tqdm import tqdm

from .chem import reaction_hash


def extract_templates(
    reactions_csv: Path,
    output_path: Path,
    *,
    radius: int = 1,
    expand_ring: bool = False,
    expand_hetero: bool = False,
    min_count: int = 3,
    limit: int | None = None,
    workers: int = 1,
) -> None:
    data = pd.read_csv(reactions_csv)
    if limit:
        data = data.head(limit)

    payloads = [
        (idx, row.to_dict(), radius, expand_ring, expand_hetero)
        for idx, row in data.iterrows()
    ]
    if workers > 1:
        with Pool(processes=workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_extract_one, payloads, chunksize=100),
                    total=len(payloads),
                    desc="extracting templates",
                )
            )
    else:
        results = [
            _extract_one(payload)
            for payload in tqdm(payloads, total=len(payloads), desc="extracting templates")
        ]

    errors = sum(1 for row in results if row is None)
    rows = [row for row in results if row is not None]
    for idx, row in enumerate(rows):
        row["index"] = idx

    counts = Counter(row["template_hash"] for row in rows)
    rows = [row for row in rows if counts[row["template_hash"]] >= min_count]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"wrote {len(rows)} template rows to {output_path}; extraction errors={errors}")


def _extract_one(
    payload: tuple[int, dict[str, Any], int, bool, bool],
) -> dict[str, object] | None:
    idx, row, radius, expand_ring, expand_hetero = payload
    reactants = str(row.get("reactants", ""))
    products = str(row.get("products", ""))
    mapped = str(row.get("mapped_smiles", row.get("reaction_smiles", "")))
    try:
        rxn = ChemicalReaction(mapped, clean_smiles=False)
        _, retro_template = rxn.generate_reaction_template(
            radius=radius,
            expand_ring=expand_ring,
            expand_hetero=expand_hetero,
        )
        smarts = retro_template.smarts
        template_hash = retro_template.hash_from_bits()
    except (ReactionException, Exception):
        return None
    if not smarts:
        return None
    return {
        "index": -1,
        "ID": row.get("id", idx),
        "reaction_hash": reaction_hash(reactants, products),
        "reactants": reactants,
        "products": products,
        "classification": row.get("classification", "-"),
        "retro_template": smarts,
        "template_hash": template_hash,
        "selectivity": "-",
        "outcomes": 1,
        "template_code": -1,
    }
