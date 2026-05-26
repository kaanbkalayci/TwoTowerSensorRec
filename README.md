## Sensor-Subset Experiments

This repo is a local experimentation workspace for selecting small sensor
subsets near a vehicle using synchronized RSSI, acoustic waveform features, and
sensor geometry. The current modeling target is a two-tower MLP that ranks all
sensor subsets of size `<= 3` at each 200 ms timestep.

The vehicle GPS is ground truth. It is used to build labels and evaluation
metrics, but it is kept out of feature vectors.

## Current Pipeline

1. Put raw FLAC, vehicle GPS, and sensor GPS files in `data/raw/`.
2. Run `notebooks/01_dataloader_smoke_test.ipynb` to verify 200 ms alignment,
   visualize gaps/RSSI/distance, extract audio features, and build baseline
   processed artifacts.
3. Run `notebooks/02_two_tower_model.ipynb` to train and compare baseline
   two-tower models and architecture sweeps.
4. Run `notebooks/03_expanded_feature_two_tower.ipynb` to build the expanded
   feature set, train the current `h512 d2 e8/e16` models, and compare against
   an RSSI+geometry-only ablation.

## Repo Structure

```text
.
├── data/
│   ├── raw/           # Local raw data: FLAC, vehicle GPS, sensor GPS
│   ├── interim/       # Optional temporary transformed data
│   ├── processed/     # Local model-ready artifacts, ignored by Git
│   └── external/      # Optional third-party/reference data
├── notebooks/         # Executable analysis and training notebooks
├── scripts/           # Reusable loaders, feature builders, training helpers
├── experiments/       # Local run outputs and lightweight experiment notes
├── reports/           # Generated figures/tables/writeups
├── docs/              # Method notes and design decisions
├── src/               # Reserved for packaged/stable project code
└── tests/             # Tests and smoke checks for reusable code
```

The `.reference-MarkovianBandits/` directory is a local ignored reference clone
used for comparison. It is not part of this project package.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU training, confirm PyTorch sees CUDA in a notebook:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Data Policy

Raw, interim, processed, and run-output files are intentionally ignored by Git.
Commit code, notebooks, README files, small schemas, and notes. Keep large data,
trained checkpoints, embeddings, and generated tables/figures local unless there
is an explicit reason to publish a small artifact.

## Main Artifacts

The processed builders write a consistent set of files with a prefix such as
`vehicle_sensor_subset_200ms`, `vehicle_sensor_subset_200ms_expanded_features`,
or `vehicle_sensor_subset_200ms_rssi_geometry_only`:

- `*_arrays.npz`: `C_by_time`, `A_examples`, labels, time ids.
- `*_meta.json`: feature names, dimensions, action layout, subset universe.
- `*_examples_index.csv`: subset metadata and ground-truth utility fields.
- `*_sequence.pkl` and `*_examples.pkl`: full context/action examples.
- `*_ground_truth_*.csv`: vehicle/node ground truth kept separate from features.

## Current Modeling Notes

- Actions are all subsets of available sensors with size `1`, `2`, or `3`.
- The saved utility is the current primary target:
  `1/(1+d1/rho) + 0.45/(1+d2/rho) + 0.20/(1+d3/rho)`.
- Decision metrics are evaluated per timestep:
  `top1`, `top3`, `mean_rank`, `avg_regret`, and `avg_norm_regret`.
- Expanded waveform features substantially outperform RSSI+geometry-only
  features in the current runs.
