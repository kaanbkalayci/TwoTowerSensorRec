from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from IPython.display import display
except Exception:  # pragma: no cover - only used outside notebooks
    display = print

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover - dependency/reporting path
    sf = None
    print("soundfile unavailable; refined-band variants will be skipped:", repr(exc))


PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_tower_training import (  # noqa: E402
    TrainConfig,
    TwoTowerMLP,
    build_utility_labels,
    compare_on_common_objectives,
    evaluate_split,
    make_loader,
    make_loss,
    prepare_standardized_data,
    set_all_seeds,
    summarize_results,
)


# ============================================================
# Experiment knobs
# ============================================================

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
BEST_DIR = PROJECT_ROOT / "experiments" / "static_action_two_tower" / "bestmodel_v2"
FEATURE_SEARCH_MODE = os.getenv("FEATURE_SEARCH_MODE", "search").strip().lower()
MODE_OUTPUTS = {
    "search": ("feature_search_two_tower", "feature_search", "fs"),
    "feature_search": ("feature_search_two_tower", "feature_search", "fs"),
    "ablation": ("feature_ablation_two_tower", "feature_ablation", "fa"),
    "ablations": ("feature_ablation_two_tower", "feature_ablation", "fa"),
    "candidate": ("feature_candidate_two_tower", "feature_candidate", "fc"),
    "candidates": ("feature_candidate_two_tower", "feature_candidate", "fc"),
    "candidate_d": ("feature_candidate_d_two_tower", "feature_candidate_d", "fd"),
    "d": ("feature_candidate_d_two_tower", "feature_candidate_d", "fd"),
    "bestmodel_v3": ("bestmodel_v3_candidate_training", "bestmodel_v3_candidate", "v3"),
}
if FEATURE_SEARCH_MODE not in MODE_OUTPUTS:
    raise ValueError(
        "FEATURE_SEARCH_MODE must be one of "
        f"{sorted(MODE_OUTPUTS)}, got {FEATURE_SEARCH_MODE!r}"
    )
EXPERIMENT_NAME, OUTPUT_STEM, RUN_PREFIX = MODE_OUTPUTS[FEATURE_SEARCH_MODE]
OUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME
TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / EXPERIMENT_NAME
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / EXPERIMENT_NAME
for path in [OUT_DIR, TABLE_DIR, FIG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

PREFIX = "vehicle_sensor_subset_200ms_expanded_features"
SAMPLE_MS = 200
REGRET_REL_TOL = 0.03
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# This is a search notebook, so epochs are shorter than the final v2 run by default.
# Increase SEARCH_MAX_EPOCHS before a final candidate bake-off.
SEARCH_MAX_EPOCHS = int(os.getenv("FEATURE_SEARCH_MAX_EPOCHS", "60"))
SEARCH_PATIENCE = int(os.getenv("FEATURE_SEARCH_PATIENCE", "60"))
LOG_EVERY = int(os.getenv("FEATURE_SEARCH_LOG_EVERY", "5"))
BATCH_SIZE = 8192
SEED = 22
RUN_REFINED_EXTRACTION = True

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DEVICE:", DEVICE)
print("FEATURE_SEARCH_MODE:", FEATURE_SEARCH_MODE)
print("OUT_DIR:", OUT_DIR)
print("SEARCH_MAX_EPOCHS:", SEARCH_MAX_EPOCHS)


# ============================================================
# Load source artifacts
# ============================================================

arrays_path = PROCESSED_DIR / f"{PREFIX}_arrays.npz"
meta_path = PROCESSED_DIR / f"{PREFIX}_meta.json"
examples_path = PROCESSED_DIR / f"{PREFIX}_examples_index.csv"
base_node_features_path = PROCESSED_DIR / f"{PREFIX}_node_features.csv"
best_info_path = BEST_DIR / "bestmodel_v2_info.json"
best_embeddings_path = BEST_DIR / "static_action_embeddings.npz"

for path in [
    arrays_path,
    meta_path,
    examples_path,
    base_node_features_path,
    best_info_path,
    best_embeddings_path,
]:
    if not path.exists():
        raise FileNotFoundError(path)

arrays = np.load(arrays_path, allow_pickle=True)
with open(meta_path, "r", encoding="utf-8") as f:
    base_meta = json.load(f)
with open(best_info_path, "r", encoding="utf-8") as f:
    best_info = json.load(f)

examples_index = pd.read_csv(examples_path)
examples_index["subset_str"] = examples_index["subset_str"].astype(str)
node_features = pd.read_csv(base_node_features_path)
node_features["datetime"] = pd.to_datetime(node_features["datetime"])
node_features["node"] = node_features["node"].astype(int)

C_base = arrays["C_by_time"].astype(np.float32)
y_examples = arrays["y_examples"].astype(np.float32)
example_time_id = arrays["example_time_id"].astype(np.int64)
sequence_times = pd.to_datetime(arrays["sequence_times"])
ordered_nodes = [int(n) for n in base_meta["ordered_nodes"]]
context_names_base = list(base_meta["context_feature_names"])
context_index = {name: i for i, name in enumerate(context_names_base)}

static_npz = np.load(best_embeddings_path, allow_pickle=True)
static_feature_names_full = [str(x) for x in static_npz["static_action_feature_names"]]
A_static_catalog_full = static_npz["action_vectors"].astype(np.float32)
subset_key = np.asarray(static_npz["subset_key"]).astype(str)
subset_to_static_idx = {name: i for i, name in enumerate(subset_key)}

if C_base.shape[1] != len(context_names_base):
    raise RuntimeError("Base context metadata does not match C_by_time.")
if A_static_catalog_full.shape[1] != len(static_feature_names_full):
    raise RuntimeError("Static action metadata does not match action catalog.")
missing_subset_labels = set(examples_index["subset_str"].unique()) - set(subset_to_static_idx)
if missing_subset_labels:
    missing = sorted(missing_subset_labels)
    raise RuntimeError(f"Subset labels missing from static action catalog: {missing[:5]}")

example_action_idx = examples_index["subset_str"].map(subset_to_static_idx).to_numpy(dtype=np.int64)
A_static_examples_full = A_static_catalog_full[example_action_idx]

print("Loaded base data")
print("  C_base:", C_base.shape)
print("  A_static_full:", A_static_examples_full.shape)
print("  examples:", len(examples_index))
print("  times:", len(sequence_times))


# ============================================================
# Feature grouping and selection helpers
# ============================================================

def parse_context_feature(name: str) -> tuple[int | None, str, str]:
    match = re.match(r"^n(\d+)_(.+)$", name)
    if not match:
        return None, name, name
    node = int(match.group(1))
    local = match.group(2)
    base = local[len("audio_") :] if local.startswith("audio_") else local
    return node, local, base


def context_group(name: str) -> str:
    node, local, base = parse_context_feature(name)
    if node is None:
        if name.startswith("rssi_") or name.startswith("global_rssi"):
            return "global RSSI summaries"
        if name.startswith("acoustic_com"):
            return "acoustic COM"
        if any(token in name for token in ["vehicle_like", "construction_like"]):
            return "global signature scores"
        if name.startswith("global_"):
            return "global audio summaries"
        return "other global"

    if base == "rssi_db":
        return "RSSI current"
    if base in {
        "rssi_mean",
        "rssi_std",
        "rssi_median",
        "rssi_mad",
        "rssi_iqr",
        "rssi_trimmed_mean",
        "rssi_slope",
        "rssi_delta1",
        "rssi_persistence_top3",
        "rssi_persistence_top5",
    }:
        return "RSSI temporal/robust"
    if base in {"energy_share", "rank_percentile"}:
        return "RSSI rank/share"
    if base in {"sensor_x_norm", "sensor_y_norm"}:
        return "sensor geometry"
    if base in {
        "vehicle_like_score",
        "construction_like_score",
        "vehicle_minus_construction_score",
    }:
        return "audio signature scores"
    if local.startswith("audio_"):
        if base in {
            "rms_db",
            "peak_db",
            "band_20_120_ratio",
            "band_120_500_ratio",
            "band_500_2000_ratio",
            "band_20_120_db",
            "band_120_500_db",
            "band_500_2000_db",
            "band_2000_6000_db",
        }:
            return "audio energy/bands"
        if base in {"low_to_voice_db", "mid_to_voice_db", "rms_delta_db", "centroid_delta_hz"}:
            return "audio ratios/deltas"
        return "audio spectral/disturbance"
    return "other per-node"


def select_existing_context(
    *,
    keep_groups: set[str] | None = None,
    drop_groups: set[str] | None = None,
    keep_bases: set[str] | None = None,
    drop_bases: set[str] | None = None,
) -> list[str]:
    out = []
    for name in context_names_base:
        _node, _local, base = parse_context_feature(name)
        group = context_group(name)
        if keep_groups is not None and group not in keep_groups:
            continue
        if drop_groups is not None and group in drop_groups:
            continue
        if keep_bases is not None and base not in keep_bases:
            continue
        if drop_bases is not None and base in drop_bases:
            continue
        out.append(name)
    return out


def select_action_names(kind: str) -> list[str]:
    masks = [name for name in static_feature_names_full if name.startswith("mask_n")]
    slots = [
        name
        for name in static_feature_names_full
        if re.match(r"^slot\d+_sensor_[xy]_norm$", name)
    ]
    centroid = ["subset_centroid_x_norm", "subset_centroid_y_norm"]
    if kind == "full20":
        return list(static_feature_names_full)
    if kind == "core15":
        return masks + slots + centroid + ["subset_size"]
    if kind == "min13":
        return masks + slots + ["subset_size"]
    raise ValueError(f"Unknown action feature set: {kind}")


# ============================================================
# Refined audio-band extraction
# ============================================================

REFINED_BANDS = [
    (20, 80),
    (80, 160),
    (160, 400),
    (400, 900),
    (900, 2000),
    (2000, 3500),
    (3500, 6000),
]
REFINED_BAND_FEATURES = [
    f"band_{lo}_{hi}_{suffix}" for lo, hi in REFINED_BANDS for suffix in ("db", "ratio")
]
REFINED_BAND_DB_FEATURES = [f"band_{lo}_{hi}_db" for lo, hi in REFINED_BANDS]
REFINED_BAND_RATIO_FEATURES = [f"band_{lo}_{hi}_ratio" for lo, hi in REFINED_BANDS]
REFINED_LOW_MID_BAND_FEATURES = [
    f"band_{lo}_{hi}_{suffix}"
    for lo, hi in REFINED_BANDS
    if hi <= 2000
    for suffix in ("db", "ratio")
]
REFINED_BAND_DELTA_FEATURES = [f"band_{lo}_{hi}_delta_db" for lo, hi in REFINED_BANDS]
REFINED_LOW_BAND_DELTA_FEATURES = [
    f"band_{lo}_{hi}_delta_db" for lo, hi in REFINED_BANDS if hi <= 900
]
REFINED_EXTRA_AUDIO_FEATURES = ["low_high_ratio_db", "voice_band_ratio", "high_band_ratio"]
REFINED_ALL_FEATURES = (
    REFINED_BAND_FEATURES + REFINED_BAND_DELTA_FEATURES + REFINED_EXTRA_AUDIO_FEATURES
)
REFINED_CACHE = PROCESSED_DIR / "vehicle_sensor_subset_200ms_refined_audio_bands_long.csv"
EPS = np.finfo(np.float32).eps


def parse_flac_start(file_path: Path, node_id: int) -> datetime | None:
    match = re.match(
        rf"(\d{{8}})_(\d{{6}})_.*_{node_id}_respeaker\.flac$",
        file_path.name,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y%m%d %H%M%S")


def to_respeaker_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    hi = min(audio.shape[1], 5)
    lo = 1 if audio.shape[1] > 1 else 0
    return audio[:, lo:hi].mean(axis=1).astype(np.float32, copy=False)


def band_power(power: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return np.full(power.shape[0], np.nan, dtype=np.float32)
    return power[:, mask].sum(axis=1)


def refined_frame_features(audio: np.ndarray, samplerate: int, sample_ms: int) -> pd.DataFrame:
    frame_len = int(round((sample_ms / 1000.0) * samplerate))
    n_frames = len(audio) // frame_len
    if n_frames <= 0:
        return pd.DataFrame(columns=REFINED_ALL_FEATURES)

    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    centered = frames - frames.mean(axis=1, keepdims=True)
    window = np.hanning(frame_len).astype(np.float32)
    spectrum = np.fft.rfft(centered * window, axis=1)
    power = np.abs(spectrum).astype(np.float32) ** 2
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / samplerate).astype(np.float32)
    total_power = power.sum(axis=1) + EPS

    band_values = {}
    for lo, hi in REFINED_BANDS:
        band_values[(lo, hi)] = band_power(power, freqs, lo, hi)

    features = {}
    for lo, hi in REFINED_BANDS:
        p = band_values[(lo, hi)]
        features[f"band_{lo}_{hi}_db"] = 10.0 * np.log10(p + EPS)
        features[f"band_{lo}_{hi}_ratio"] = p / total_power

    low_power = band_values[(20, 80)] + band_values[(80, 160)] + EPS
    high_power = band_values[(2000, 3500)] + band_values[(3500, 6000)] + EPS
    voice_power = (
        band_values[(400, 900)]
        + band_values[(900, 2000)]
        + band_values[(2000, 3500)]
        + EPS
    )
    features["low_high_ratio_db"] = 10.0 * np.log10(low_power / high_power)
    features["voice_band_ratio"] = voice_power / total_power
    features["high_band_ratio"] = high_power / total_power

    df = pd.DataFrame(features)
    for lo, hi in REFINED_BANDS:
        name = f"band_{lo}_{hi}_db"
        df[f"band_{lo}_{hi}_delta_db"] = df[name].diff().fillna(0.0)
    return df[REFINED_ALL_FEATURES]


def build_refined_audio_long() -> pd.DataFrame | None:
    if not RUN_REFINED_EXTRACTION:
        return None
    if REFINED_CACHE.exists():
        cached = pd.read_csv(REFINED_CACHE)
        if set(["datetime", "node", *REFINED_ALL_FEATURES]).issubset(cached.columns):
            cached["datetime"] = pd.to_datetime(cached["datetime"])
            cached["node"] = cached["node"].astype(int)
            print("Loaded refined audio cache:", REFINED_CACHE, cached.shape)
            return cached

    if sf is None:
        warnings.warn("soundfile is not available; refined-band variants will be skipped.")
        return None

    rows = []
    for node in ordered_nodes:
        flacs = sorted(RAW_DIR.glob(f"*_{node}_respeaker.flac"))
        if not flacs:
            warnings.warn(f"No FLAC found for node {node}; refined-band variants will be skipped.")
            return None
        node_parts = []
        for flac in flacs:
            start_time = parse_flac_start(flac, node)
            if start_time is None:
                continue
            audio, samplerate = sf.read(flac, dtype="float32")
            mono = to_respeaker_mono(audio)
            feat = refined_frame_features(mono, samplerate, SAMPLE_MS)
            feat.insert(
                0,
                "datetime",
                pd.date_range(start=start_time, periods=len(feat), freq=f"{SAMPLE_MS}ms"),
            )
            feat.insert(1, "node", int(node))
            node_parts.append(feat)
            print(f"refined audio: node {node}, {flac.name}, rows={len(feat)}, sr={samplerate}")
        if node_parts:
            rows.append(pd.concat(node_parts, ignore_index=True))

    if not rows:
        return None
    refined = pd.concat(rows, ignore_index=True)
    refined = (
        refined.sort_values(["datetime", "node"])
        .drop_duplicates(["datetime", "node"], keep="first")
        .reset_index(drop=True)
    )
    refined.to_csv(REFINED_CACHE, index=False)
    print("Saved refined audio cache:", REFINED_CACHE, refined.shape)
    return refined


refined_audio_long = build_refined_audio_long()


def matrix_from_refined_features(feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    if not feature_names:
        return np.zeros((len(sequence_times), 0), dtype=np.float32), []
    if refined_audio_long is None:
        raise RuntimeError("Refined audio was requested, but refined_audio_long is unavailable.")

    mats = []
    names = []
    audio = refined_audio_long.copy()
    audio["datetime"] = pd.to_datetime(audio["datetime"])
    audio["node"] = audio["node"].astype(int)
    sequence_index = pd.DatetimeIndex(sequence_times)

    for node in ordered_nodes:
        node_df = audio.loc[audio["node"].eq(node)].set_index("datetime")
        for feat in feature_names:
            if feat not in node_df.columns:
                raise KeyError(f"Missing refined feature {feat}")
            values = node_df[feat].reindex(sequence_index).to_numpy(dtype=float)
            if np.isfinite(values).any():
                fill = float(np.nanmedian(values[np.isfinite(values)]))
            else:
                fill = 0.0
            mats.append(np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill).astype(np.float32))
            names.append(f"n{node}_audio_{feat}")
    return np.column_stack(mats).astype(np.float32), names


# ============================================================
# Variant definitions
# ============================================================

SAFE_DROP_GROUPS = {"global signature scores", "global RSSI summaries", "acoustic COM"}
SAFE_DROP_BASES = {"rssi_db"}

RSSI_CORE_BASES = {
    "rssi_std",
    "rssi_slope",
    "rssi_persistence_top3",
    "rssi_persistence_top5",
    "energy_share",
    "rank_percentile",
    "sensor_x_norm",
    "sensor_y_norm",
}
SPECTRAL_CORE_BASES = {"spectral_flatness", "spectral_entropy", "zcr"}
SPECTRAL_EXTENDED_BASES = SPECTRAL_CORE_BASES | {
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff85_hz",
}
OLD_BAND_BASES = {
    "band_20_120_db",
    "band_120_500_db",
    "band_500_2000_db",
    "band_2000_6000_db",
    "band_20_120_ratio",
    "band_120_500_ratio",
    "band_500_2000_ratio",
    "band_2000_6000_ratio",
}
MINIMAL_RSSI_GEOM_BASES = {
    "rssi_persistence_top3",
    "rssi_persistence_top5",
    "energy_share",
    "rank_percentile",
    "sensor_x_norm",
    "sensor_y_norm",
}
MINIMAL_SENSOR_GEOM_BASES = {"sensor_x_norm", "sensor_y_norm"}
MINIMAL_BASES = MINIMAL_RSSI_GEOM_BASES | SPECTRAL_CORE_BASES

SEARCH_VARIANTS = [
    {
        "name": "safe_pruned_existing",
        "description": (
            "Drop groups that mean-ablation did not hurt: raw RSSI current, global RSSI, "
            "acoustic COM, global signature scores."
        ),
        "existing_context": select_existing_context(
            drop_groups=SAFE_DROP_GROUPS,
            drop_bases=SAFE_DROP_BASES,
        ),
        "refined_audio": [],
        "action_set": "full20",
    },
    {
        "name": "safe_no_handcrafted_scores",
        "description": "Safe pruning plus remove per-node handcrafted vehicle/construction scores.",
        "existing_context": select_existing_context(
            drop_groups=SAFE_DROP_GROUPS | {"audio signature scores"},
            drop_bases=SAFE_DROP_BASES,
        ),
        "refined_audio": [],
        "action_set": "full20",
    },
    {
        "name": "aggressive_existing_audio_core",
        "description": (
            "Keep compact RSSI rank/temporal core, old broad audio bands, extended spectral "
            "features, and sensor geometry."
        ),
        "existing_context": select_existing_context(
            keep_bases=RSSI_CORE_BASES | OLD_BAND_BASES | SPECTRAL_EXTENDED_BASES
        ),
        "refined_audio": [],
        "action_set": "core15",
    },
    {
        "name": "refined_bands_core",
        "description": (
            "Replace broad bands with refined band db/ratio features; keep compact RSSI core "
            "and cheap spectral features."
        ),
        "existing_context": select_existing_context(keep_bases=RSSI_CORE_BASES | SPECTRAL_CORE_BASES),
        "refined_audio": REFINED_BAND_FEATURES,
        "action_set": "core15",
    },
    {
        "name": "refined_bands_plus_deltas",
        "description": "Refined bands plus per-band temporal deltas and simple low/high/voice ratios.",
        "existing_context": select_existing_context(keep_bases=RSSI_CORE_BASES | SPECTRAL_CORE_BASES),
        "refined_audio": REFINED_BAND_FEATURES
        + REFINED_BAND_DELTA_FEATURES
        + REFINED_EXTRA_AUDIO_FEATURES,
        "action_set": "core15",
    },
    {
        "name": "minimal_deploy_refined",
        "description": (
            "Small deployment candidate: persistence/share RSSI, sensor geometry, refined "
            "bands, and three spectral disturbance features."
        ),
        "existing_context": select_existing_context(
            keep_bases=MINIMAL_BASES
        ),
        "refined_audio": REFINED_BAND_FEATURES,
        "action_set": "min13",
    },
]

ABLATION_VARIANTS = [
    {
        "name": "minimal_deploy_refined_reference",
        "description": "Same compact deployment candidate as the first-cell winner; included as the ablation anchor.",
        "existing_context": select_existing_context(keep_bases=MINIMAL_BASES),
        "refined_audio": REFINED_BAND_FEATURES,
        "action_set": "min13",
    },
    {
        "name": "ablate_no_spectral_core",
        "description": "Remove spectral_flatness, spectral_entropy, and zcr; keep RSSI/share/rank, sensor geometry, and refined bands.",
        "existing_context": select_existing_context(keep_bases=MINIMAL_RSSI_GEOM_BASES),
        "refined_audio": REFINED_BAND_FEATURES,
        "action_set": "min13",
    },
    {
        "name": "ablate_top3_persistence_only",
        "description": "Remove rssi_persistence_top5 to test whether top3 persistence is enough.",
        "existing_context": select_existing_context(
            keep_bases=(MINIMAL_BASES - {"rssi_persistence_top5"})
        ),
        "refined_audio": REFINED_BAND_FEATURES,
        "action_set": "min13",
    },
    {
        "name": "ablate_no_energy_share",
        "description": "Remove energy_share while keeping rank_percentile and persistence features.",
        "existing_context": select_existing_context(keep_bases=(MINIMAL_BASES - {"energy_share"})),
        "refined_audio": REFINED_BAND_FEATURES,
        "action_set": "min13",
    },
    {
        "name": "ablate_no_rank_percentile",
        "description": "Remove rank_percentile while keeping energy_share and persistence features.",
        "existing_context": select_existing_context(keep_bases=(MINIMAL_BASES - {"rank_percentile"})),
        "refined_audio": REFINED_BAND_FEATURES,
        "action_set": "min13",
    },
    {
        "name": "ablate_refined_db_only",
        "description": "Use refined band dB powers only; drop the ratio version of each refined band.",
        "existing_context": select_existing_context(keep_bases=MINIMAL_BASES),
        "refined_audio": REFINED_BAND_DB_FEATURES,
        "action_set": "min13",
    },
    {
        "name": "ablate_refined_ratio_only",
        "description": "Use refined band ratios only; drop absolute dB powers.",
        "existing_context": select_existing_context(keep_bases=MINIMAL_BASES),
        "refined_audio": REFINED_BAND_RATIO_FEATURES,
        "action_set": "min13",
    },
    {
        "name": "ablate_low_mid_bands_only",
        "description": "Keep refined bands through 2 kHz only; remove 2-3.5 kHz and 3.5-6 kHz disturbance-sensitive bands.",
        "existing_context": select_existing_context(keep_bases=MINIMAL_BASES),
        "refined_audio": REFINED_LOW_MID_BAND_FEATURES,
        "action_set": "min13",
    },
    {
        "name": "add_low_high_ratio",
        "description": "Add one cheap low-vs-high spectral ratio to the minimal refined feature set.",
        "existing_context": select_existing_context(keep_bases=MINIMAL_BASES),
        "refined_audio": REFINED_BAND_FEATURES + ["low_high_ratio_db"],
        "action_set": "min13",
    },
    {
        "name": "add_low_band_deltas",
        "description": "Add temporal deltas only for the low/mid vehicle bands through 900 Hz.",
        "existing_context": select_existing_context(keep_bases=MINIMAL_BASES),
        "refined_audio": REFINED_BAND_FEATURES + REFINED_LOW_BAND_DELTA_FEATURES,
        "action_set": "min13",
    },
    {
        "name": "add_rssi_temporal",
        "description": "Add rssi_std and rssi_slope back into the minimal refined feature set.",
        "existing_context": select_existing_context(
            keep_bases=MINIMAL_BASES | {"rssi_std", "rssi_slope"}
        ),
        "refined_audio": REFINED_BAND_FEATURES,
        "action_set": "min13",
    },
]

CANDIDATE_VARIANTS = [
    {
        "name": "drop_rssi_keep_spectral_db_ratio",
        "description": (
            "Candidate A: drop RSSI persistence/share/rank context features; keep sensor "
            "geometry, spectral core, and refined band dB plus ratio features."
        ),
        "existing_context": select_existing_context(
            keep_bases=MINIMAL_SENSOR_GEOM_BASES | SPECTRAL_CORE_BASES
        ),
        "refined_audio": REFINED_BAND_FEATURES,
        "action_set": "min13",
    },
    {
        "name": "drop_rssi_drop_spectral_db_ratio",
        "description": (
            "Candidate B: drop RSSI persistence/share/rank and spectral core; keep sensor "
            "geometry and refined band dB plus ratio features."
        ),
        "existing_context": select_existing_context(keep_bases=MINIMAL_SENSOR_GEOM_BASES),
        "refined_audio": REFINED_BAND_FEATURES,
        "action_set": "min13",
    },
    {
        "name": "drop_rssi_keep_spectral_db_only",
        "description": (
            "Candidate C: drop RSSI persistence/share/rank; keep sensor geometry, spectral "
            "core, and refined band dB features only."
        ),
        "existing_context": select_existing_context(
            keep_bases=MINIMAL_SENSOR_GEOM_BASES | SPECTRAL_CORE_BASES
        ),
        "refined_audio": REFINED_BAND_DB_FEATURES,
        "action_set": "min13",
    },
]

CANDIDATE_D_VARIANTS = [
    {
        "name": "drop_rssi_drop_spectral_db_only",
        "description": (
            "Candidate D: drop RSSI persistence/share/rank and spectral core; keep sensor "
            "geometry and refined band dB features only."
        ),
        "existing_context": select_existing_context(keep_bases=MINIMAL_SENSOR_GEOM_BASES),
        "refined_audio": REFINED_BAND_DB_FEATURES,
        "action_set": "min13",
    },
]

if FEATURE_SEARCH_MODE in {"search", "feature_search"}:
    VARIANTS = SEARCH_VARIANTS
elif FEATURE_SEARCH_MODE in {"ablation", "ablations"}:
    VARIANTS = ABLATION_VARIANTS
elif FEATURE_SEARCH_MODE in {"candidate", "candidates"}:
    VARIANTS = CANDIDATE_VARIANTS
elif FEATURE_SEARCH_MODE in {"candidate_d", "d", "bestmodel_v3"}:
    VARIANTS = CANDIDATE_D_VARIANTS
else:  # Guarded by MODE_OUTPUTS above.
    raise ValueError(f"Unhandled FEATURE_SEARCH_MODE {FEATURE_SEARCH_MODE!r}")

variant_filter = os.getenv("FEATURE_SEARCH_ONLY", "").strip()
if variant_filter:
    requested_variants = {name.strip() for name in variant_filter.split(",") if name.strip()}
    known_variants = {variant["name"] for variant in VARIANTS}
    missing_variants = sorted(requested_variants - known_variants)
    if missing_variants:
        raise ValueError(f"Unknown FEATURE_SEARCH_ONLY variants: {missing_variants}")
    VARIANTS = [variant for variant in VARIANTS if variant["name"] in requested_variants]

if refined_audio_long is None:
    VARIANTS = [variant for variant in VARIANTS if not variant["refined_audio"]]
    print("Refined audio unavailable; running only existing-feature variants.")

print("Planned variants:")
for variant in VARIANTS:
    print(
        f"  {variant['name']}: existing={len(variant['existing_context'])}, "
        f"refined={len(variant['refined_audio'])}, action={variant['action_set']}"
    )


# ============================================================
# Dataset construction
# ============================================================

def make_context_matrix(existing_names: list[str], refined_features: list[str]) -> tuple[np.ndarray, list[str]]:
    pieces = []
    names = []
    if existing_names:
        idx = [context_index[name] for name in existing_names]
        pieces.append(C_base[:, idx].astype(np.float32))
        names.extend(existing_names)
    if refined_features:
        refined_mat, refined_names = matrix_from_refined_features(refined_features)
        pieces.append(refined_mat)
        names.extend(refined_names)
    if not pieces:
        raise ValueError("Variant has no context features.")
    return np.column_stack(pieces).astype(np.float32), names


def make_action_examples(action_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    idx = [static_feature_names_full.index(name) for name in action_names]
    a_catalog = A_static_catalog_full[:, idx].astype(np.float32)
    a_examples = a_catalog[example_action_idx].astype(np.float32)
    return a_examples, a_catalog, action_names


def make_variant_data(variant: dict) -> tuple[dict, np.ndarray, dict]:
    c_variant, context_names = make_context_matrix(variant["existing_context"], variant["refined_audio"])
    action_names = select_action_names(variant["action_set"])
    a_examples, a_catalog, action_names = make_action_examples(action_names)

    meta = copy.deepcopy(base_meta)
    meta["context_feature_names"] = context_names
    meta["context_dim"] = int(c_variant.shape[1])
    meta["action_feature_names"] = action_names
    meta["action_raw_dim"] = int(a_examples.shape[1])
    meta["static_action_feature_names"] = action_names
    meta["feature_search_variant"] = variant["name"]
    meta["feature_search_description"] = variant["description"]
    meta["refined_audio_features_added"] = list(variant["refined_audio"])
    meta["source_prefix"] = PREFIX
    meta["leakage_note"] = base_meta.get("leakage_note", "")

    data = {
        "C_by_time": c_variant,
        "A_examples": a_examples,
        "y_examples": y_examples.copy(),
        "saved_y_examples": y_examples.copy(),
        "example_time_id": example_time_id.copy(),
        "sequence_times": np.asarray(sequence_times, dtype="datetime64[ns]"),
        "examples_index": examples_index.copy(),
        "meta": meta,
        "utility_name": "saved",
        "utility_kwargs": {},
    }
    manifest = {
        "variant": variant["name"],
        "description": variant["description"],
        "context_dim": int(c_variant.shape[1]),
        "action_dim": int(a_examples.shape[1]),
        "action_set": variant["action_set"],
        "context_feature_names": context_names,
        "static_action_feature_names": action_names,
        "refined_audio_features_added": list(variant["refined_audio"]),
    }
    return data, a_catalog, manifest


# ============================================================
# Training with bestmodel_v2-style epoch selection
# ============================================================

def selector_key(val: dict[str, float], best_regret_so_far: float) -> tuple[float, ...]:
    regret = float(val["avg_regret"])
    regret_floor = best_regret_so_far * (1.0 + REGRET_REL_TOL)
    regret_penalty = max(0.0, regret - regret_floor)
    return (
        regret_penalty,
        float(val["mean_rank"]),
        -float(val["top3"]),
        -float(val["top1"]),
        float(val["avg_norm_regret"]),
        float(val["rmse"]),
    )


def train_variant(data: dict, a_catalog: np.ndarray, manifest: dict, config: TrainConfig) -> dict:
    set_all_seeds(config.seed)
    run_dir = OUT_DIR / config.run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_data = dict(data)
    if train_data.get("utility_name") != config.utility_name or train_data.get("utility_kwargs") != (
        config.utility_kwargs or {}
    ):
        train_data["y_examples"] = build_utility_labels(
            train_data["examples_index"],
            train_data["saved_y_examples"],
            train_data["meta"],
            config.utility_name,
            config.utility_kwargs or {},
        ).astype(np.float32)
        train_data["utility_name"] = config.utility_name
        train_data["utility_kwargs"] = config.utility_kwargs or {}

    prepared = prepare_standardized_data(train_data, config)
    train_loader = make_loader(prepared, prepared["split"]["train"], config, shuffle=True)

    model = TwoTowerMLP(
        context_dim=prepared["C_by_time"].shape[1],
        action_dim=prepared["A_examples"].shape[1],
        hidden=config.hidden,
        emb_dim=config.emb_dim,
        depth=config.depth,
        dropout=config.dropout,
        combine_mode=config.combine_mode,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = make_loss(config.loss_name)

    best_state = None
    best_key = None
    best_epoch = -1
    best_regret_so_far = float("inf")
    wait = 0
    history_rows = []
    start = time.time()

    print(f"\n=== {config.run_name} ===")
    print(
        f"context_dim={prepared['C_by_time'].shape[1]} "
        f"action_dim={prepared['A_examples'].shape[1]} "
        f"desc={manifest['description']}"
    )

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for c_batch, a_batch, y_batch in train_loader:
            c_batch = c_batch.to(DEVICE)
            a_batch = a_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(c_batch, a_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(losses))
        val = evaluate_split(model, prepared, "val", config, DEVICE)
        best_regret_so_far = min(best_regret_so_far, float(val["avg_regret"]))
        key = selector_key(val, best_regret_so_far)
        improved = best_key is None or key < best_key
        if improved:
            best_key = key
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            torch.save(
                {
                    "epoch": int(epoch),
                    "selector_key": [float(x) for x in key],
                    "best_regret_so_far": float(best_regret_so_far),
                    "val_metrics": {k: float(v) for k, v in val.items()},
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "context_dim": int(prepared["C_by_time"].shape[1]),
                    "action_dim": int(prepared["A_examples"].shape[1]),
                    "standardizers": prepared["standardizers"],
                    "meta": prepared["meta"],
                    "manifest": manifest,
                    "selection_rule": (
                        f"avg_regret within {REGRET_REL_TOL:.1%} of best-so-far, "
                        "then mean_rank, top3, top1, avg_norm_regret, rmse"
                    ),
                },
                ckpt_dir / "best_by_regret_tolerant_rank.pt",
            )
        else:
            wait += 1

        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "best_regret_so_far": float(best_regret_so_far),
                "selector_regret_penalty": float(key[0]),
                "selector_mean_rank": float(key[1]),
                "selector_neg_top3": float(key[2]),
                "selector_neg_top1": float(key[3]),
                "is_best_epoch_so_far": bool(improved),
                **{f"val_{k}": float(v) for k, v in val.items()},
            }
        )

        if epoch == 1 or epoch % config.log_every == 0 or improved or epoch == config.max_epochs:
            star = "*" if improved else " "
            print(
                f"{config.run_name} ep {epoch:03d}/{config.max_epochs:03d}{star} "
                f"loss={train_loss:.5f} val_reg={val['avg_regret']:.4f} "
                f"rank={val['mean_rank']:.2f} top1={val['top1']:.3f} "
                f"top3={val['top3']:.3f} rmse={val['rmse']:.4f} "
                f"{time.time() - start:.0f}s"
            )

        if wait >= config.patience:
            print(f"early stop at epoch {epoch}; selected epoch {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"No best state selected for {config.run_name}")

    model.load_state_dict(best_state)
    metrics = {
        split: evaluate_split(model, prepared, split, config, DEVICE)
        for split in ("train", "val", "test")
    }

    result = {
        "config": config,
        "model": model,
        "prepared": prepared,
        "history": pd.DataFrame(history_rows),
        "metrics": metrics,
        "best_epoch": int(best_epoch),
        "device": DEVICE,
    }

    summary_df = summarize_results([result])
    common_eval_df = compare_on_common_objectives(
        [result],
        objectives=[
            {"name": "saved_rational", "utility_name": "saved", "utility_kwargs": {}},
            {"name": "contains_closest", "utility_name": "closest_binary", "utility_kwargs": {}},
            {"name": "rank_discount", "utility_name": "rank_discount", "utility_kwargs": {}},
        ],
        split_names=("train", "val", "test"),
    )

    a_mu = prepared["standardizers"]["A_mu"]
    a_sigma = prepared["standardizers"]["A_sigma"]
    a_sigma = np.where(np.abs(a_sigma) < 1e-8, 1.0, a_sigma).astype(np.float32)
    a_catalog_std = ((a_catalog - a_mu) / a_sigma).astype(np.float32)
    model.eval()
    with torch.no_grad():
        a_tensor = torch.from_numpy(a_catalog_std).to(DEVICE)
        static_action_embeddings = model.embed_action(a_tensor).detach().cpu().numpy().astype(np.float32)

    checkpoint_path = run_dir / "selected_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "context_dim": int(prepared["C_by_time"].shape[1]),
            "action_dim": int(prepared["A_examples"].shape[1]),
            "standardizers": prepared["standardizers"],
            "meta": prepared["meta"],
            "best_epoch": int(best_epoch),
            "best_key": [float(x) for x in best_key],
            "regret_rel_tol": float(REGRET_REL_TOL),
            "metrics": metrics,
            "manifest": manifest,
        },
        checkpoint_path,
    )
    np.savez_compressed(
        run_dir / "static_action_embeddings.npz",
        action_embeddings=static_action_embeddings,
        action_vectors=a_catalog.astype(np.float32),
        action_vectors_std=a_catalog_std.astype(np.float32),
        subset_key=subset_key,
        static_action_feature_names=np.asarray(manifest["static_action_feature_names"], dtype=object),
    )
    result["history"].to_csv(run_dir / "history.csv", index=False)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    common_eval_df.to_csv(run_dir / "common_eval.csv", index=False)
    with open(run_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    result["run_dir"] = run_dir
    result["summary_df"] = summary_df
    result["common_eval_df"] = common_eval_df
    result["manifest"] = manifest
    return result


# ============================================================
# Run search
# ============================================================

base_cfg = TrainConfig(
    run_name="placeholder",
    utility_name="saved",
    hidden=512,
    emb_dim=16,
    depth=2,
    dropout=0.05,
    combine_mode="mul_only",
    loss_name="mse",
    lr=5e-4,
    weight_decay=1e-4,
    batch_size=BATCH_SIZE,
    max_epochs=SEARCH_MAX_EPOCHS,
    patience=SEARCH_PATIENCE,
    seed=SEED,
    train_frac=0.60,
    val_frac=0.20,
    log_every=LOG_EVERY,
    num_workers=0,
)

all_results = []
variant_manifests = []

for variant in VARIANTS:
    data_variant, a_catalog_variant, manifest = make_variant_data(variant)
    run_name = f"{RUN_PREFIX}_{variant['name']}_h512_d2_e16"
    cfg = copy.deepcopy(base_cfg)
    cfg.run_name = run_name
    result = train_variant(data_variant, a_catalog_variant, manifest, cfg)
    all_results.append(result)
    variant_manifests.append(manifest)


# ============================================================
# Compare against bestmodel_v2 and save outputs
# ============================================================

summary_df = (
    pd.concat([result["summary_df"] for result in all_results], ignore_index=True)
    if all_results
    else pd.DataFrame()
)
common_eval_df = (
    pd.concat([result["common_eval_df"] for result in all_results], ignore_index=True)
    if all_results
    else pd.DataFrame()
)

rows = []
best_common_path = BEST_DIR / "common_eval.csv"
if best_common_path.exists():
    best_common = pd.read_csv(best_common_path)
    best_contains = best_common.query("eval_objective == 'contains_closest' and split == 'test'").iloc[0]
    best_saved = best_common.query("eval_objective == 'saved_rational' and split == 'test'").iloc[0]
    rows.append(
        {
            "run_name": "bestmodel_v2_reference",
            "variant": "bestmodel_v2_reference",
            "context_dim": int(best_info["context_dim"]),
            "action_dim": int(best_info["action_dim"]),
            "total_input_dim": int(best_info["context_dim"] + best_info["action_dim"]),
            "best_epoch": int(best_info["best_epoch"]),
            "test_contains_closest": float(best_contains["top1"]),
            "test_saved_top1": float(best_saved["top1"]),
            "test_saved_top3": float(best_saved["top3"]),
            "test_mean_rank": float(best_saved["mean_rank"]),
            "test_avg_regret": float(best_saved["avg_regret"]),
            "test_avg_norm_regret": float(best_saved["avg_norm_regret"]),
            "description": "Current bestmodel_v2 reference; static action tower, full expanded context.",
        }
    )

for result in all_results:
    manifest = result["manifest"]
    common_eval = result["common_eval_df"]
    contains = common_eval.query("eval_objective == 'contains_closest' and split == 'test'").iloc[0]
    saved = common_eval.query("eval_objective == 'saved_rational' and split == 'test'").iloc[0]
    rows.append(
        {
            "run_name": result["config"].run_name,
            "variant": manifest["variant"],
            "context_dim": int(manifest["context_dim"]),
            "action_dim": int(manifest["action_dim"]),
            "total_input_dim": int(manifest["context_dim"] + manifest["action_dim"]),
            "best_epoch": int(result["best_epoch"]),
            "test_contains_closest": float(contains["top1"]),
            "test_saved_top1": float(saved["top1"]),
            "test_saved_top3": float(saved["top3"]),
            "test_mean_rank": float(saved["mean_rank"]),
            "test_avg_regret": float(saved["avg_regret"]),
            "test_avg_norm_regret": float(saved["avg_norm_regret"]),
            "description": manifest["description"],
        }
    )

comparison_df = pd.DataFrame(rows)
if not comparison_df.empty:
    reference_acc = float(
        comparison_df.loc[
            comparison_df["variant"].eq("bestmodel_v2_reference"),
            "test_contains_closest",
        ].iloc[0]
    )
    reference_dim = float(
        comparison_df.loc[
            comparison_df["variant"].eq("bestmodel_v2_reference"),
            "total_input_dim",
        ].iloc[0]
    )
    comparison_df["contains_delta_vs_v2"] = comparison_df["test_contains_closest"] - reference_acc
    comparison_df["feature_reduction_vs_v2"] = 1.0 - comparison_df["total_input_dim"] / reference_dim
    comparison_df = comparison_df.sort_values(
        ["test_contains_closest", "feature_reduction_vs_v2", "test_avg_regret"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

summary_path = TABLE_DIR / f"{OUTPUT_STEM}_summary.csv"
common_eval_path = TABLE_DIR / f"{OUTPUT_STEM}_common_eval.csv"
comparison_path = TABLE_DIR / f"{OUTPUT_STEM}_comparison.csv"
manifest_path = TABLE_DIR / f"{OUTPUT_STEM}_variant_manifests.json"
summary_df.to_csv(summary_path, index=False)
common_eval_df.to_csv(common_eval_path, index=False)
comparison_df.to_csv(comparison_path, index=False)
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(variant_manifests, f, indent=2)

if not comparison_df.empty:
    display(comparison_df)
    candidates = comparison_df.loc[~comparison_df["variant"].eq("bestmodel_v2_reference")].copy()
    candidates["meets_half_point_rule"] = candidates["contains_delta_vs_v2"] >= -0.005
    if not candidates.empty:
        best_candidate = candidates.sort_values(
            [
                "meets_half_point_rule",
                "test_contains_closest",
                "feature_reduction_vs_v2",
                "test_avg_regret",
            ],
            ascending=[False, False, False, True],
        ).iloc[0]
        print("\nBest candidate by <=0.5 pp tolerance, then accuracy/reduction/regret:")
        print(best_candidate.to_string())

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    plot_df = comparison_df.copy()
    ax.scatter(
        100.0 * plot_df["feature_reduction_vs_v2"],
        100.0 * plot_df["test_contains_closest"],
        s=70,
        color=[
            "#222222" if variant == "bestmodel_v2_reference" else "#2f6f9f"
            for variant in plot_df["variant"]
        ],
    )
    for row in plot_df.itertuples(index=False):
        ax.annotate(
            row.variant,
            (100.0 * row.feature_reduction_vs_v2, 100.0 * row.test_contains_closest),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(100.0 * reference_acc, color="0.4", linestyle="--", linewidth=0.9)
    ax.axhline(100.0 * (reference_acc - 0.005), color="0.65", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Input feature reduction vs bestmodel_v2 (%)")
    ax.set_ylabel("Test closest-sensor containment (%)")
    ax.set_title(f"{OUTPUT_STEM.replace('_', ' ').title()}: accuracy vs input size")
    ax.grid(alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{OUTPUT_STEM}_accuracy_vs_feature_reduction.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{OUTPUT_STEM}_accuracy_vs_feature_reduction.png", dpi=300, bbox_inches="tight")
    plt.show()

print("\nSaved:")
for path in [
    summary_path,
    common_eval_path,
    comparison_path,
    manifest_path,
    FIG_DIR / f"{OUTPUT_STEM}_accuracy_vs_feature_reduction.pdf",
]:
    print(" ", path)
