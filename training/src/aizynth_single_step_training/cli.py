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

DEFAULT_RUN_NAME = "retrocast_v2026-05-12_ss_reaction-holdout-n1-n5"
DEFAULT_WORK_DIR = Path("runs") / DEFAULT_RUN_NAME


@click.group()
def main() -> None:
    """train an aizynthfinder-compatible single-step expansion policy."""


@main.command()
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--artifact", default="single-step-reaction-holdout-n1-n5", show_default=True)
@click.option("--split", multiple=True, default=("training", "validation"), show_default=True)
@click.option("--format", "wire_format", default="jsonl", type=click.Choice(["jsonl", "rsmi"]), show_default=True)
@click.option("--dry-run", is_flag=True)
def download_retrocast(
    output_dir: Path, artifact: str, split: tuple[str, ...], wire_format: str, dry_run: bool
) -> None:
    """download hosted retrocast training-set artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    script = "https://files.ischemist.com/retrocast/get-training-set.sh"
    for split_name in split:
        click.echo(f"downloading {artifact} split={split_name} into {output_dir}")
        dry = " --dry-run" if dry_run else ""
        subprocess.run(
            [
                "bash",
                "-lc",
                f"curl -fsSL {script} | bash -s -- {artifact} --split={split_name} --format={wire_format} --dir={output_dir}{dry}",
            ],
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
    with _open_text(input_path) as infile, output_path.open("w") as outfile:
        outfile.write("reaction_smiles,reactants,products,mapped_smiles\n")
        for line in infile:
            line = line.strip()
            if not line:
                continue
            mapped = line
            if line.startswith("{"):
                record = json.loads(line)
                mapped = (
                    record.get("mapped_smiles")
                    or record.get("reaction_smiles")
                    or record.get("smiles")
                    or ""
                )
                reactants, products = _split_reaction_smiles(mapped)
                reactants = reactants or _join_smiles(record.get("reactants"))
                products = products or _join_smiles(record.get("product") or record.get("products"))
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


@main.command()
@click.option("--work-dir", type=click.Path(path_type=Path), default=DEFAULT_WORK_DIR, show_default=True)
@click.option("--file-prefix", default=DEFAULT_RUN_NAME, show_default=True)
@click.option("--epochs", default=100, show_default=True)
@click.option("--batch-size", default=256, show_default=True)
@click.option("--hidden-nodes", default=512, show_default=True)
@click.option("--dropout", default=0.4, show_default=True)
@click.option("--fit-verbose", default=2, type=click.Choice(["0", "1", "2"]), show_default=True)
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


def _split_reaction_smiles(smiles: str) -> tuple[str, str]:
    parts = smiles.split(">")
    if len(parts) >= 3:
        return parts[0], parts[2]
    return "", ""


def _join_smiles(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ".".join(str(item) for item in value)
    return str(value)


def _csv(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
