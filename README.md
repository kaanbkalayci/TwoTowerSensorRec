# MILCOM Data Experiments

Workspace for data exploration, experiments, and reproducible analysis.

## Structure

```text
.
├── data/
│   ├── raw/           # Original, immutable data drops
│   ├── interim/       # Intermediate transformed data
│   ├── processed/     # Cleaned/model-ready datasets
│   └── external/      # Third-party reference data
├── notebooks/         # Exploratory notebooks
├── src/               # Reusable project code
├── scripts/           # One-off or repeatable command-line workflows
├── experiments/       # Experiment configs, notes, and outputs
├── reports/           # Figures, tables, and written outputs
├── docs/              # Project notes and methodology
└── tests/             # Tests for reusable code
```

## Data Policy

Keep large or sensitive datasets out of Git. Commit small metadata files,
schemas, sample data, and scripts that recreate derived datasets.

Recommended flow:

1. Put original files in `data/raw/`.
2. Write cleaning or feature scripts in `scripts/` or reusable logic in `src/`.
3. Save derived outputs to `data/interim/` or `data/processed/`.
4. Capture exploratory work in `notebooks/`.
5. Store final figures and tables in `reports/`.

