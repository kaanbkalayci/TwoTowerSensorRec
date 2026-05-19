# Experiments

This directory is for lightweight experiment notes, configs, and local run
outputs.

## Runs

`experiments/runs/` is ignored by Git except for `.gitkeep`. The two-tower
notebook writes checkpoints, histories, metrics JSON, and frozen embeddings
there when `export_frozen_embeddings(...)` is called.

Expected export files per run:

- `*_checkpoint.pt`: model state, config, dimensions, standardizers, metadata.
- `*_embeddings.npz`: context/action embeddings and scores.
- `*_history.csv`: epoch diagnostics.
- `*_metrics.json`: train/val/test metrics.

Keep run outputs local unless a small, curated artifact is needed for reporting.
