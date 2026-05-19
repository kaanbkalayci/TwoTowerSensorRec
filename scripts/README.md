# Scripts

Reusable project code lives here. The notebooks import these modules directly by
adding `scripts/` to `sys.path`.

## Modules

- `dataloader.py`
  - Loads raw FLAC-derived RSSI power, vehicle GPS, and sensor GPS.
  - Projects GPS to a local metric CRS.
  - Interpolates vehicle GPS and aligns data to a 200 ms grid.
  - Provides compact gap/RSSI/distance plotting helpers.

- `audio_feature_dataloader.py`
  - Extracts per-node waveform features from FLAC files at 200 ms.
  - Features include RMS, peak, crest factor, ZCR, spectral centroid/bandwidth,
    rolloff, flatness, entropy, band powers/ratios, and deltas.
  - Does not read vehicle GPS, so it is safe for feature extraction.

- `processed_feature_builder.py`
  - Builds leakage-free context/action vectors for two-tower training.
  - Creates all sensor subsets of size `<= 3`.
  - Saves features separately from vehicle/node ground truth.
  - Supports baseline, expanded waveform, and RSSI+geometry-only feature sets.

- `two_tower_training.py`
  - Defines `TrainConfig`, `TwoTowerMLP`, data loading, chronological splits,
    training loops, shared-objective evaluation, and embedding export.
  - Supported utilities include `saved`, `rational`, `clipped_linear`,
    `closest_binary`, and `rank_discount`.

- `__init__.py`
  - Re-exports the main helpers for convenience.

## Leakage Rule

Feature vectors may use sensor GPS, RSSI history/statistics, waveform features,
and subset geometry. They must not contain vehicle coordinates, distance-to-
vehicle columns, or closest-sensor labels. Those belong only in labels,
ground-truth files, and evaluation metadata.
