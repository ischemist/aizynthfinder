from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .chem import smiles_to_fingerprint
from .config import TrainingConfig


def preprocess_expansion(template_library: Path, config: TrainingConfig) -> None:
    config.output_path.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(template_library)
    data = data.drop_duplicates(subset="reaction_hash")
    counts = data.groupby("template_hash").size().sort_values(ascending=False)
    kept = counts[counts >= config.template_occurrence].index
    data = data[data["template_hash"].isin(kept)].copy()

    encoder = LabelEncoder()
    data["template_code"] = encoder.fit_transform(data["template_hash"])
    data.to_csv(config.filename("library"), index=False)

    labels = sparse.csr_matrix(
        (
            np.ones(len(data), dtype=np.int8),
            (np.arange(len(data)), data["template_code"].to_numpy()),
        ),
        shape=(len(data), data["template_code"].nunique()),
    )
    inputs = sparse.vstack(
        [
            sparse.csr_matrix(
                smiles_to_fingerprint(smiles, config.fingerprint_radius, config.fingerprint_len)
            )
            for smiles in tqdm(data["products"].to_numpy(), desc="fingerprints")
        ],
        format="csr",
    )

    _split_and_save(inputs, "inputs", config)
    _split_and_save(labels, "labels", config)
    _split_and_save(data, "library", config)
    _save_unique_templates(data, config)


def preprocess_expansion_splits(
    training_template_library: Path,
    validation_template_library: Path,
    config: TrainingConfig,
) -> None:
    config.output_path.mkdir(parents=True, exist_ok=True)
    train_data = pd.read_csv(training_template_library).drop_duplicates(
        subset="reaction_hash"
    )
    val_data = pd.read_csv(validation_template_library).drop_duplicates(
        subset="reaction_hash"
    )

    counts = train_data.groupby("template_hash").size().sort_values(ascending=False)
    kept = counts[counts >= config.template_occurrence].index
    train_data = train_data[train_data["template_hash"].isin(kept)].copy()
    val_data = val_data[val_data["template_hash"].isin(kept)].copy()

    encoder = LabelEncoder()
    train_data["template_code"] = encoder.fit_transform(train_data["template_hash"])
    code_lookup = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    val_data["template_code"] = val_data["template_hash"].map(code_lookup).astype(int)

    train_data.to_csv(config.filename("training_library"), index=False)
    val_data.to_csv(config.filename("validation_library"), index=False)
    val_data.iloc[0:0].to_csv(config.filename("testing_library"), index=False)
    pd.concat([train_data, val_data], ignore_index=True).to_csv(
        config.filename("library"), index=False
    )

    _save_matrix_split(train_data, "training", config, len(encoder.classes_))
    _save_matrix_split(val_data, "validation", config, len(encoder.classes_))
    _save_matrix_split(val_data.iloc[0:0], "testing", config, len(encoder.classes_))
    _save_unique_templates(train_data, config)


def preprocess_expansion_all(template_library: Path, config: TrainingConfig) -> None:
    config.output_path.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(template_library).drop_duplicates(subset="reaction_hash")

    counts = data.groupby("template_hash").size().sort_values(ascending=False)
    kept = counts[counts >= config.template_occurrence].index
    data = data[data["template_hash"].isin(kept)].copy()

    encoder = LabelEncoder()
    data["template_code"] = encoder.fit_transform(data["template_hash"])

    data.to_csv(config.filename("all_library"), index=False)
    data.to_csv(config.filename("training_library"), index=False)
    data.iloc[0:0].to_csv(config.filename("validation_library"), index=False)
    data.iloc[0:0].to_csv(config.filename("testing_library"), index=False)
    data.to_csv(config.filename("library"), index=False)

    _save_matrix_split(data, "training", config, len(encoder.classes_))
    _save_matrix_split(data.iloc[0:0], "validation", config, len(encoder.classes_))
    _save_matrix_split(data.iloc[0:0], "testing", config, len(encoder.classes_))
    _save_unique_templates(data, config)


def _split_and_save(data, label: str, config: TrainingConfig) -> None:
    train_size = config.split_size["training"]
    testing_frac = config.split_size["testing"]
    validation_frac = config.split_size["validation"]
    testing_size = testing_frac / (testing_frac + validation_frac)
    train_arr, test_arr = train_test_split(data, train_size=train_size, random_state=42, shuffle=True)
    val_arr, test_arr = train_test_split(test_arr, test_size=testing_size, random_state=42, shuffle=True)
    for prefix, arr in {"training_": train_arr, "validation_": val_arr, "testing_": test_arr}.items():
        filename = config.filename(prefix + label)
        if isinstance(arr, pd.DataFrame):
            arr.to_csv(filename, index=False)
        else:
            sparse.save_npz(filename, arr, compressed=True)


def _save_matrix_split(
    data: pd.DataFrame, split: str, config: TrainingConfig, nlabels: int
) -> None:
    labels = sparse.csr_matrix(
        (
            np.ones(len(data), dtype=np.int8),
            (np.arange(len(data)), data["template_code"].to_numpy(dtype=int)),
        ),
        shape=(len(data), nlabels),
    )
    if len(data):
        inputs = sparse.vstack(
            [
                sparse.csr_matrix(
                    smiles_to_fingerprint(
                        smiles, config.fingerprint_radius, config.fingerprint_len
                    )
                )
                for smiles in tqdm(
                    data["products"].to_numpy(), desc=f"{split} fingerprints"
                )
            ],
            format="csr",
        )
    else:
        inputs = sparse.csr_matrix((0, config.fingerprint_len), dtype=np.int8)
    sparse.save_npz(config.filename(f"{split}_inputs"), inputs, compressed=True)
    sparse.save_npz(config.filename(f"{split}_labels"), labels, compressed=True)


def _save_unique_templates(data: pd.DataFrame, config: TrainingConfig) -> None:
    template_group = data.groupby("template_hash", sort=False).size()
    unique = data[
        ["retro_template", "template_code", "template_hash", "classification"]
    ].drop_duplicates(subset="template_code", keep="first")
    unique["classification"] = unique["classification"].fillna("-")
    unique["library_occurrence"] = template_group.values
    unique = unique.set_index("template_code").sort_index()
    try:
        unique.to_hdf(config.filename("unique_templates_hdf5"), key="table")
    except ImportError:
        print("pytables is not installed; skipped hdf5 template export")
    unique.to_csv(config.filename("unique_templates_csv"), sep="\t", compression="gzip")
