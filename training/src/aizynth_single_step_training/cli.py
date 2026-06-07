from __future__ import annotations

import gzip
import json
import os
import subprocess
from pathlib import Path

import click

from .config import TrainingConfig


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")

DEFAULT_RELEASE = "v2026-06-05"
DEFAULT_ARTIFACT = "single-step-reaction-holdout-n1-n5"
DEFAULT_RUN_NAME = f"retrocast_{DEFAULT_RELEASE}_ss_reaction-holdout-n1-n5"
DEFAULT_WORK_DIR = Path("runs") / DEFAULT_RUN_NAME


@click.group()
def main() -> None:
    """train an aizynthfinder-compatible single-step expansion policy."""


@main.command()
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--artifact", default=DEFAULT_ARTIFACT, show_default=True)
@click.option("--release", default=DEFAULT_RELEASE, show_default=True)
@click.option(
    "--split",
    multiple=True,
    default=("training", "validation"),
    type=click.Choice(["training", "validation", "all"]),
    show_default=True,
)
@click.option("--format", "wire_format", default="rsmi", type=click.Choice(["jsonl", "rsmi"]), show_default=True)
@click.option("--dry-run", is_flag=True)
def download_retrocast(
    output_dir: Path,
    artifact: str,
    release: str,
    split: tuple[str, ...],
    wire_format: str,
    dry_run: bool,
) -> None:
    """download hosted retrocast training-set artifacts with the retrocast cli."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in split:
        click.echo(
            f"downloading {artifact} release={release} split={split_name} "
            f"format={wire_format} into {output_dir}"
        )
        command = [
            "retrocast",
            "get-training-data",
            artifact,
            "--release",
            release,
            "--split",
            split_name,
            "--format",
            wire_format,
            "--dir",
            str(output_dir),
        ]
        if dry_run:
            command.append("--dry-run")
        subprocess.run(
            command,
            check=True,
        )


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--limit", type=int, default=None)
def normalize_reactions(input_path: Path, output_path: Path, limit: int | None) -> None:
    """convert retrocast jsonl/rsmi reaction files into product/reactants csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open_text(input_path) as infile, _open_write_text(output_path) as outfile:
        outfile.write("reaction_smiles,reactants,products,mapped_smiles\n")
        for line in infile:
            line = line.strip()
            if not line:
                continue
            mapped = line
            if line.startswith("{"):
                record = json.loads(line)
                mapped = (
                    _first_string_value(
                        record,
                        (
                            "mapped_smiles",
                            "mapped_reaction_smiles",
                            "reaction_smiles",
                            "reaction_smarts",
                            "smiles",
                        ),
                    )
                    or ""
                )
                reactants, products = _split_reaction_smiles(mapped)
                reactants = reactants or _join_smiles(
                    _first_value(record, ("reactants", "precursors"))
                )
                products = products or _join_smiles(
                    _first_value(record, ("product", "products", "target"))
                )
            else:
                reactants, products = _split_reaction_smiles(line)
            if not mapped:
                continue
            if not reactants or not products:
                reactants, products = _split_reaction_smiles(mapped)
            outfile.write(
                f"{_csv(mapped)},{_csv(reactants)},{_csv(products)},{_csv(mapped)}\n"
            )
            count += 1
            if limit and count >= limit:
                break
    click.echo(f"wrote {count} reactions to {output_path}")


@main.command("combine-rsmi")
@click.argument("input_paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--keep-duplicates", is_flag=True)
@click.option("--limit", type=int, default=None)
def combine_rsmi(
    input_paths: tuple[Path, ...],
    output_path: Path,
    keep_duplicates: bool,
    limit: int | None,
) -> None:
    """combine retrocast rsmi/jsonl reaction files into one rsmi text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total = 0
    written = 0
    with _open_write_text(output_path) as outfile:
        for input_path in input_paths:
            with _open_text(input_path) as infile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    reaction_smiles = _reaction_smiles_from_line(line)
                    if not reaction_smiles:
                        continue
                    if not keep_duplicates and reaction_smiles in seen:
                        continue
                    seen.add(reaction_smiles)
                    outfile.write(reaction_smiles + "\n")
                    written += 1
                    if limit and written >= limit:
                        click.echo(
                            f"wrote {written} reactions to {output_path}; "
                            f"read={total}; duplicates_skipped={total - written}"
                        )
                        return
    click.echo(
        f"wrote {written} reactions to {output_path}; "
        f"read={total}; duplicates_skipped={total - written}"
    )


@main.command()
@click.argument("reactions_csv", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--radius", default=1, show_default=True)
@click.option("--expand-ring", is_flag=True)
@click.option("--expand-hetero", is_flag=True)
@click.option("--min-count", default=1, show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("--workers", default=1, show_default=True)
def extract(
    reactions_csv: Path,
    output_path: Path,
    radius: int,
    expand_ring: bool,
    expand_hetero: bool,
    min_count: int,
    limit: int | None,
    workers: int,
) -> None:
    """extract rxnutils/rdchiral retro templates from normalized reactions."""
    from .templates import extract_templates

    extract_templates(
        reactions_csv,
        output_path,
        radius=radius,
        expand_ring=expand_ring,
        expand_hetero=expand_hetero,
        min_count=min_count,
        limit=limit,
        workers=workers,
    )


@main.command()
@click.argument("template_library", type=click.Path(exists=True, path_type=Path))
@click.option("--work-dir", type=click.Path(path_type=Path), default=DEFAULT_WORK_DIR, show_default=True)
@click.option("--file-prefix", default=DEFAULT_RUN_NAME, show_default=True)
@click.option("--template-occurrence", default=3, show_default=True)
@click.option("--fingerprint-len", default=2048, show_default=True)
@click.option("--fingerprint-radius", default=2, show_default=True)
def preprocess(
    template_library: Path,
    work_dir: Path,
    file_prefix: str,
    template_occurrence: int,
    fingerprint_len: int,
    fingerprint_radius: int,
) -> None:
    """build sparse matrices and unique template table."""
    from .preprocess import preprocess_expansion

    config = TrainingConfig(
        output_path=work_dir,
        file_prefix=file_prefix,
        template_occurrence=template_occurrence,
        fingerprint_len=fingerprint_len,
        fingerprint_radius=fingerprint_radius,
    )
    preprocess_expansion(template_library, config)


@main.command("preprocess-splits")
@click.argument("training_template_library", type=click.Path(exists=True, path_type=Path))
@click.argument("validation_template_library", type=click.Path(exists=True, path_type=Path))
@click.option("--work-dir", type=click.Path(path_type=Path), default=DEFAULT_WORK_DIR, show_default=True)
@click.option("--file-prefix", default=DEFAULT_RUN_NAME, show_default=True)
@click.option("--template-occurrence", default=3, show_default=True)
@click.option("--fingerprint-len", default=2048, show_default=True)
@click.option("--fingerprint-radius", default=2, show_default=True)
def preprocess_splits(
    training_template_library: Path,
    validation_template_library: Path,
    work_dir: Path,
    file_prefix: str,
    template_occurrence: int,
    fingerprint_len: int,
    fingerprint_radius: int,
) -> None:
    """preprocess fixed train/validation template libraries without re-splitting."""
    from .preprocess import preprocess_expansion_splits

    config = TrainingConfig(
        output_path=work_dir,
        file_prefix=file_prefix,
        template_occurrence=template_occurrence,
        fingerprint_len=fingerprint_len,
        fingerprint_radius=fingerprint_radius,
    )
    preprocess_expansion_splits(
        training_template_library, validation_template_library, config
    )


@main.command("preprocess-all")
@click.argument("template_library", type=click.Path(exists=True, path_type=Path))
@click.option("--work-dir", type=click.Path(path_type=Path), default=DEFAULT_WORK_DIR, show_default=True)
@click.option("--file-prefix", default=DEFAULT_RUN_NAME, show_default=True)
@click.option("--template-occurrence", default=3, show_default=True)
@click.option("--fingerprint-len", default=2048, show_default=True)
@click.option("--fingerprint-radius", default=2, show_default=True)
def preprocess_all(
    template_library: Path,
    work_dir: Path,
    file_prefix: str,
    template_occurrence: int,
    fingerprint_len: int,
    fingerprint_radius: int,
) -> None:
    """preprocess one production template library using every retained row for training."""
    from .preprocess import preprocess_expansion_all

    config = TrainingConfig(
        output_path=work_dir,
        file_prefix=file_prefix,
        template_occurrence=template_occurrence,
        fingerprint_len=fingerprint_len,
        fingerprint_radius=fingerprint_radius,
    )
    preprocess_expansion_all(template_library, config)


@main.command()
@click.option("--work-dir", type=click.Path(path_type=Path), default=DEFAULT_WORK_DIR, show_default=True)
@click.option("--file-prefix", default=DEFAULT_RUN_NAME, show_default=True)
@click.option("--epochs", default=100, show_default=True)
@click.option("--batch-size", default=256, show_default=True)
@click.option("--hidden-nodes", default=512, show_default=True)
@click.option("--dropout", default=0.4, show_default=True)
@click.option("--fit-verbose", default=0, type=click.Choice(["0", "1", "2"]), show_default=True)
def train(
    work_dir: Path,
    file_prefix: str,
    epochs: int,
    batch_size: int,
    hidden_nodes: int,
    dropout: float,
    fit_verbose: str,
) -> None:
    """train the original-style keras expansion network."""
    from .train import train_expansion_model

    config = TrainingConfig(
        output_path=work_dir,
        file_prefix=file_prefix,
        epochs=epochs,
        batch_size=batch_size,
        hidden_nodes=hidden_nodes,
        drop_out=dropout,
        fit_verbose=int(fit_verbose),
    )
    train_expansion_model(config)


@main.command()
@click.option("--work-dir", type=click.Path(path_type=Path), default=DEFAULT_WORK_DIR, show_default=True)
@click.option("--file-prefix", default=DEFAULT_RUN_NAME, show_default=True)
@click.option("--model", "model_path", type=click.Path(path_type=Path), default=None)
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=Path("aizynth_config.yml"))
def write_config(work_dir: Path, file_prefix: str, model_path: Path | None, output_path: Path) -> None:
    """write an aizynthfinder config snippet for the trained policy."""
    model = model_path or work_dir / "checkpoints" / "keras_model.hdf5"
    templates = work_dir / f"{file_prefix}_unique_templates.csv.gz"
    output_path.write_text(
        "expansion:\n"
        f"  {file_prefix}:\n"
        "    type: template-based\n"
        f"    model: {model}\n"
        f"    template: {templates}\n"
        "    template_column: retro_template\n"
        "    cutoff_cumulative: 0.995\n"
        "    cutoff_number: 50\n",
        encoding="utf8",
    )
    click.echo(f"wrote {output_path}")


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf8") if path.suffix == ".gz" else path.open(encoding="utf8")


def _open_write_text(path: Path):
    return gzip.open(path, "wt", encoding="utf8") if path.suffix == ".gz" else path.open("w", encoding="utf8")


def _reaction_smiles_from_line(line: str) -> str:
    if line.startswith("{"):
        record = json.loads(line)
        return (
            _first_string_value(
                record,
                (
                    "mapped_smiles",
                    "mapped_reaction_smiles",
                    "reaction_smiles",
                    "reaction_smarts",
                    "smiles",
                ),
            )
            or ""
        )
    return line


def _split_reaction_smiles(smiles: str) -> tuple[str, str]:
    parts = smiles.split(">")
    if len(parts) >= 3:
        return parts[0], parts[2]
    return "", ""


def _join_smiles(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ".".join(_stringify_smiles(item) for item in value)
    if isinstance(value, dict):
        return _first_string_value(
            value,
            ("mapped_smiles", "smiles", "reaction_smiles", "mapped_reaction_smiles"),
        )
    return str(value)


def _csv(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _first_value(obj, keys: tuple[str, ...]):
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] not in (None, ""):
                return obj[key]
        for value in obj.values():
            found = _first_value(value, keys)
            if found not in (None, ""):
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _first_value(value, keys)
            if found not in (None, ""):
                return found
    return None


def _first_string_value(obj, keys: tuple[str, ...]) -> str:
    value = _first_value(obj, keys)
    return _stringify_smiles(value)


def _stringify_smiles(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _first_string_value(
            value,
            ("mapped_smiles", "smiles", "reaction_smiles", "mapped_reaction_smiles"),
        )
    if isinstance(value, list):
        return ".".join(_stringify_smiles(item) for item in value)
    return str(value)
