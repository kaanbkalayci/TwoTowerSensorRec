# Notebooks

Run notebooks from the repo root when possible. Each notebook also detects when
it is launched from inside `notebooks/` and adjusts paths back to the project
root.

## Notebook Order

1. `01_dataloader_smoke_test.ipynb`
   - Loads raw FLAC/RSSI, vehicle GPS, and sensor GPS.
   - Resamples to 200 ms.
   - Visualizes trajectory, gaps, and log RSSI vs log distance.
   - Extracts audio features from FLAC files.
   - Builds and saves baseline processed two-tower artifacts.

2. `02_two_tower_model.ipynb`
   - Loads processed artifacts.
   - Trains saved/clipped/binary utility baselines.
   - Evaluates all utilities on common downstream objectives.
   - Runs architecture and seed sweeps.
   - Exports frozen embeddings/checkpoints for selected runs.

3. `03_expanded_feature_two_tower.ipynb`
   - Builds the expanded feature set and verifies every requested feature.
   - Trains the saved-utility `h512 d2 e8/e16` style models.
   - Builds and trains an RSSI+geometry-only ablation for comparison.

## Metric Reminder

Use the common-objective tables for comparing runs. Own-target tables are useful
diagnostics, but they are not apples-to-apples across utility functions.

- `top1`: the model-selected subset is truly best for the evaluation objective.
- `top3`: the selected subset is in the true top three actions.
- `mean_rank`: average true rank of the selected subset.
- `avg_regret`: utility lost against the best possible subset.
- `avg_norm_regret`: regret normalized within each timestep.
