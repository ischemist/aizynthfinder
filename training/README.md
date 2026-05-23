# aizynth single-step training

standalone `uv` project for training an aizynthfinder-compatible template expansion policy.

this intentionally clones the old aizynthfinder expansion model family:

- product morgan fingerprint, default radius `2`, length `2048`
- one hidden dense layer, `elu`, default `512` nodes
- l2 kernel regularization `0.001`
- dropout, default `0.4`
- softmax over retained templates
- categorical crossentropy with top-k metrics

## what aizynthtrain is

`aizynthtrain` was molecularai's separate pipeline repo for training synthesis prediction models consumed by aizynthfinder. it is not itself the model. its readme says it produced `uspto_keras_model.hdf5` and `uspto_unique_templates.csv.gz`, which are exactly the two expansion-policy artifacts aizynthfinder config expects. that repo is now archived and points users toward `aizynthmodels`.

## setup

```bash
cd training
uv sync --extra train
```

on linux gpu hosts with recent nvidia drivers:

```bash
cd training
uv sync --extra cuda
```

## end-to-end retrocast run

download the single-step reaction-holdout training and validation splits:

```bash
uv run aizynth-train-one-step download-retrocast \
  --artifact single-step-reaction-holdout-n1-n5 \
  --split training \
  --split validation \
  --format jsonl \
  --output-dir data/raw
```

add `--dry-run` first if you just want to see the resolved url/path without downloading.

normalize whichever downloaded `*.jsonl.gz` or `*.rsmi.txt.gz` files you want to train from:

```bash
uv run aizynth-train-one-step normalize-reactions \
  data/raw/path/to/training.jsonl.gz \
  --output data/processed/training_reactions.csv
```

extract rxnutils/rdchiral templates:

```bash
uv run aizynth-train-one-step extract \
  data/processed/training_reactions.csv \
  --output runs/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5_training_raw_template_library.csv \
  --radius 1 \
  --min-count 1 \
  --workers 8

uv run aizynth-train-one-step extract \
  data/processed/validation_reactions.csv \
  --output runs/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5_validation_raw_template_library.csv \
  --radius 1 \
  --min-count 1 \
  --workers 8
```

preprocess into sparse matrices and template tables:

```bash
uv run aizynth-train-one-step preprocess \
  runs/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5_training_raw_template_library.csv \
  --template-occurrence 3
```

if you extracted both retrocast training and validation splits separately, keep that fixed split instead:

```bash
uv run aizynth-train-one-step preprocess-splits \
  runs/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5_training_raw_template_library.csv \
  runs/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5_validation_raw_template_library.csv \
  --template-occurrence 3
```

train:

```bash
uv run aizynth-train-one-step train \
  --epochs 100 \
  --batch-size 256
```

write an aizynthfinder config snippet:

```bash
uv run aizynth-train-one-step write-config \
  --output runs/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5/aizynth_config.yml
```

the important outputs are:

- `runs/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5/checkpoints/keras_model.hdf5`
- `runs/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5/retrocast_v2026-05-12_ss_reaction-holdout-n1-n5_unique_templates.csv.gz`

## migrating old local names

if you already ran the earlier `paroutes` examples, rename the run directory and prefix-bearing files:

```bash
old=paroutes
new=retrocast_v2026-05-12_ss_reaction-holdout-n1-n5

mkdir -p "runs/$new"
find "runs/$old" -maxdepth 1 -type f -name "${old}_*" -print0 |
  while IFS= read -r -d '' path; do
    base=$(basename "$path")
    mv "$path" "runs/$new/${base/#$old/$new}"
  done

if [ -d "runs/$old/checkpoints" ]; then
  mv "runs/$old/checkpoints" "runs/$new/checkpoints"
fi
```

## notes

the template extraction stage assumes mapped reactions. if the input split is not atom-mapped, rdchiral template extraction will fail for most rows. use the retrocast `reaction_records` jsonl where possible, since it preserves mapped smiles metadata.
