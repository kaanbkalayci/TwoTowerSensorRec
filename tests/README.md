# Tests

Tests and smoke checks for reusable code belong here.

Useful near-term tests:

- `processed_feature_builder` creates context/action arrays whose dimensions
  match metadata feature-name lists.
- Expanded feature builds fail when requested audio features are missing.
- RSSI+geometry-only builds contain no `audio_*` or derived waveform features.
- Feature vectors do not include vehicle coordinates or distance-to-vehicle
  columns.
- `two_tower_training.decision_metrics` preserves the intended meanings of
  `top1`, `top3`, and `mean_rank`.

For now, small synthetic-data tests are preferred over tests that require local
FLAC/GPS files.
