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

## end-to-end paroutes / retrocast run

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

extract rdchiral templates:

```bash
uv run aizynth-train-one-step extract \
  data/processed/training_reactions.csv \
  --output runs/paroutes/paroutes_raw_template_library.csv \
  --radius 1 \
  --min-count 3 \
  --workers 8
```

preprocess into sparse matrices and template tables:

```bash
uv run aizynth-train-one-step preprocess \
  runs/paroutes/paroutes_raw_template_library.csv \
  --work-dir runs/paroutes \
  --file-prefix paroutes
```

if you extracted both retrocast training and validation splits separately, keep that fixed split instead:

```bash
uv run aizynth-train-one-step preprocess-splits \
  runs/paroutes/paroutes_training_raw_template_library.csv \
  runs/paroutes/paroutes_validation_raw_template_library.csv \
  --work-dir runs/paroutes \
  --file-prefix paroutes
```

train:

```bash
uv run aizynth-train-one-step train \
  --work-dir runs/paroutes \
  --file-prefix paroutes \
  --epochs 100 \
  --batch-size 256
```

write an aizynthfinder config snippet:

```bash
uv run aizynth-train-one-step write-config \
  --work-dir runs/paroutes \
  --file-prefix paroutes \
  --output runs/paroutes/aizynth_config.yml
```

the important outputs are:

- `runs/paroutes/checkpoints/keras_model.hdf5`
- `runs/paroutes/paroutes_unique_templates.csv.gz`

## notes

the template extraction stage assumes mapped reactions. if the input split is not atom-mapped, rdchiral template extraction will fail for most rows. use the retrocast `reaction_records` jsonl where possible, since it preserves mapped smiles metadata.
