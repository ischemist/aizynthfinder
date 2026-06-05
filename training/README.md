# aizynth single-step training

standalone `uv` project for training aizynthfinder-compatible template expansion policies from retrocast single-step training sets.

the model intentionally follows the old aizynthfinder expansion policy family:

- product morgan fingerprint, default radius `2`, length `2048`
- one hidden dense layer, `elu`, default `512` nodes
- l2 kernel regularization `0.001`
- dropout, default `0.4`
- softmax over retained templates
- categorical crossentropy with top-k metrics

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

`uv.toml` includes a project-local `retrocast` freshness exception so same-day retrocast releases work even if your global uv config has `exclude-newer = "7 days"`.

## datasets

the current release is `v2026-06-05`, using retrocast `v0.7.0` schema v2 downloads.

production runs train from `all.rsmi.txt.gz`, not the training/validation splits:

```text
retrocast_v2026-06-05_ss_reaction-holdout-n1-n5
retrocast_v2026-06-05_ss_route-holdout-n1-n5
```

source artifacts:

```text
single-step-reaction-holdout-n1-n5
single-step-route-holdout-n1-n5
```

## gpu production command

to run the full production pipeline for both models:

```bash
cd training
uv sync --extra cuda
scripts/run-production-training.sh both
```

to run one model:

```bash
scripts/run-production-training.sh reaction
scripts/run-production-training.sh route
```

override defaults with environment variables:

```bash
WORKERS=16 EPOCHS=100 BATCH_SIZE=256 TEMPLATE_OCCURRENCE=3 \
  scripts/run-production-training.sh both
```

## download

```bash
uv run aizynth-train-one-step download-retrocast \
  --release v2026-06-05 \
  --artifact single-step-reaction-holdout-n1-n5 \
  --split all \
  --format rsmi \
  --output-dir data/raw

uv run aizynth-train-one-step download-retrocast \
  --release v2026-06-05 \
  --artifact single-step-route-holdout-n1-n5 \
  --split all \
  --format rsmi \
  --output-dir data/raw
```

## reaction-holdout production run

```bash
release=v2026-06-05
artifact=single-step-reaction-holdout-n1-n5
run=retrocast_${release}_ss_reaction-holdout-n1-n5

uv run aizynth-train-one-step normalize-reactions \
  data/raw/${release}/${artifact}/all.rsmi.txt.gz \
  --output data/processed/${run}_all_reactions.csv

uv run aizynth-train-one-step extract \
  data/processed/${run}_all_reactions.csv \
  --output runs/${run}/${run}_all_raw_template_library.csv \
  --radius 1 \
  --min-count 1 \
  --workers 8

uv run aizynth-train-one-step preprocess-all \
  runs/${run}/${run}_all_raw_template_library.csv \
  --work-dir runs/${run} \
  --file-prefix ${run} \
  --template-occurrence 3

uv run aizynth-train-one-step train \
  --work-dir runs/${run} \
  --file-prefix ${run} \
  --epochs 100 \
  --batch-size 256

uv run aizynth-train-one-step write-config \
  --work-dir runs/${run} \
  --file-prefix ${run} \
  --output runs/${run}/aizynth_config.yml
```

## route-holdout production run

```bash
release=v2026-06-05
artifact=single-step-route-holdout-n1-n5
run=retrocast_${release}_ss_route-holdout-n1-n5

uv run aizynth-train-one-step normalize-reactions \
  data/raw/${release}/${artifact}/all.rsmi.txt.gz \
  --output data/processed/${run}_all_reactions.csv

uv run aizynth-train-one-step extract \
  data/processed/${run}_all_reactions.csv \
  --output runs/${run}/${run}_all_raw_template_library.csv \
  --radius 1 \
  --min-count 1 \
  --workers 8

uv run aizynth-train-one-step preprocess-all \
  runs/${run}/${run}_all_raw_template_library.csv \
  --work-dir runs/${run} \
  --file-prefix ${run} \
  --template-occurrence 3

uv run aizynth-train-one-step train \
  --work-dir runs/${run} \
  --file-prefix ${run} \
  --epochs 100 \
  --batch-size 256

uv run aizynth-train-one-step write-config \
  --work-dir runs/${run} \
  --file-prefix ${run} \
  --output runs/${run}/aizynth_config.yml
```

## outputs

each production run writes:

```text
runs/<run>/checkpoints/keras_model.hdf5
runs/<run>/checkpoints/keras_model_best_loss.keras
runs/<run>/checkpoints/keras_model_final.hdf5
runs/<run>/<run>_unique_templates.csv.gz
runs/<run>/<run>_keras_training.log
runs/<run>/aizynth_config.yml
```

`keras_model.hdf5` is exported from the best-loss checkpoint and is the default path used by `write-config`.

## notes

template extraction assumes atom-mapped reactions. prefer retrocast `rsmi` downloads for this workflow because they contain mapped reaction smiles directly.
