# Data

This directory holds local data artifacts. Most contents are ignored by Git; the
directory layout and this README are tracked.

## Layout

- `raw/`: immutable source data.
  - Expected files currently include six `*_respeaker.flac` files, one vehicle
    GPS CSV, and `wp_node_gps.csv` for sensor locations.
- `interim/`: optional temporary transforms or diagnostics.
- `processed/`: model-ready two-tower artifacts written by the notebooks.
- `external/`: optional third-party or reference datasets.

## Raw Data Expectations

The current notebooks expect:

```text
data/raw/
├── 20260416_150152_dvpg_gq_orin_11_respeaker.flac
├── ...
├── 20260416_150152_dvpg_gq_orin_16_respeaker.flac
├── 20260416_150152_gps2_gps.csv
└── wp_node_gps.csv
```

FLAC files are parsed by node id and start timestamp from the filename. The
pipeline resamples/aligns audio-derived features and interpolated vehicle GPS to
200 ms.

## Processed Prefixes

- `vehicle_sensor_subset_200ms`: baseline RSSI/audio/geometry features.
- `vehicle_sensor_subset_200ms_expanded_features`: expanded waveform, RSSI,
  global, and subset features.
- `vehicle_sensor_subset_200ms_rssi_geometry_only`: ablation without waveform
  features; keeps RSSI-derived features and sensor geometry.

Feature artifacts intentionally exclude vehicle coordinates and distances.
Ground-truth vehicle and node distance data are saved separately.
