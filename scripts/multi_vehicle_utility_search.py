from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from itertools import combinations
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from two_tower_training import TrainConfig, TwoTowerMLP, fit_standardizer, make_loss, set_all_seeds


DEFAULT_INTERIM_NAME = "20250815_0900_multi_vehicle"
DEFAULT_PREFIX = "m02_multi_vehicle_eq5_rssi_static"
DEFAULT_MODEL_TAG = "multi_vehicle_rssi_utility_search_eq5"

V3_REFINED_BANDS = (
    (20, 80),
    (80, 160),
    (160, 400),
    (400, 900),
    (900, 2000),
    (2000, 3500),
    (3500, 6000),
)
V3_REFINED_BAND_DB_FEATURES = [f"band_{lo}_{hi}_db" for lo, hi in V3_REFINED_BANDS]
FEATURE_EPS = np.finfo(np.float32).eps

UTILITY_CANDIDATES = [
    {
        "name": "sum_full",
        "column": "utility_sum_full",
        "description": "u1_full + u2_full, where full uses d1/d2/d3 terms.",
    },
    {
        "name": "sum_d1_only",
        "column": "utility_sum_d1_only",
        "description": "f(d1_vehicle1) + f(d1_vehicle2).",
    },
    {
        "name": "min_d1_only",
        "column": "utility_min_d1_only",
        "description": "min(f(d1_vehicle1), f(d1_vehicle2)); protects the worse-covered vehicle.",
    },
    {
        "name": "balanced_d1",
        "column": "utility_balanced_d1",
        "description": "0.5*mean(d1 utilities) + 0.5*min(d1 utilities).",
    },
    {
        "name": "balanced_full",
        "column": "utility_balanced_full",
        "description": "0.5*mean(full utilities) + 0.5*min(full utilities).",
    },
]

SECOND_SWEEP_CANDIDATES = [
    {
        "name": "softmin_d1_beta5",
        "column": "derived_softmin_d1_beta5",
        "description": "Smooth min over per-vehicle d1 utilities with beta=5.",
        "kind": "softmin_d1",
        "beta": 5.0,
    },
    {
        "name": "softmin_d1_beta10",
        "column": "derived_softmin_d1_beta10",
        "description": "Sharper smooth min over per-vehicle d1 utilities with beta=10.",
        "kind": "softmin_d1",
        "beta": 10.0,
    },
    {
        "name": "sum_plus_softmin_beta10_lam05",
        "column": "derived_sum_plus_softmin_beta10_lam05",
        "description": "mean(d1 utilities) + 0.5 * softmin_beta10(d1 utilities).",
        "kind": "sum_plus_softmin_d1",
        "beta": 10.0,
        "lam": 0.5,
    },
    {
        "name": "sum_plus_softmin_beta10_lam10",
        "column": "derived_sum_plus_softmin_beta10_lam10",
        "description": "mean(d1 utilities) + 1.0 * softmin_beta10(d1 utilities).",
        "kind": "sum_plus_softmin_d1",
        "beta": 10.0,
        "lam": 1.0,
    },
    {
        "name": "harmonic_d1",
        "column": "derived_harmonic_d1",
        "description": "Harmonic mean of per-vehicle d1 utilities.",
        "kind": "harmonic_d1",
    },
    {
        "name": "sum_d1_minus_gap_lam025",
        "column": "derived_sum_d1_minus_gap_lam025",
        "description": "sum(d1 utilities) - 0.25 * range(d1 utilities), discouraging vehicle imbalance.",
        "kind": "sum_d1_minus_gap",
        "lam": 0.25,
    },
    {
        "name": "top2_d1_alpha02",
        "column": "derived_top2_d1_alpha02",
        "description": "For each vehicle, f(d1) + 0.2*f(d2), then sum over vehicles.",
        "kind": "top2_d1",
        "alpha": 0.2,
    },
]

ALL_UTILITY_CANDIDATES = UTILITY_CANDIDATES + SECOND_SWEEP_CANDIDATES


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    root = Path(project_root or Path.cwd())
    if root.name == "notebooks":
        root = root.parent
    return root


def _parse_flac_start(path: Path, node: int) -> datetime:
    match = re.match(rf"(\d{{8}})_(\d{{6}})_.*_{node}_respeaker\.flac$", path.name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse FLAC start time from {path.name}")
    return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y%m%d %H%M%S")


def _to_respeaker_mono(block: np.ndarray) -> np.ndarray:
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)
    hi = min(block.shape[1], 5)
    lo = 1 if block.shape[1] > 1 else 0
    return block[:, lo:hi].mean(axis=1).astype(np.float32, copy=False)


def _band_power(power: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return np.zeros(power.shape[0], dtype=np.float32)
    return power[:, mask].sum(axis=1).astype(np.float32)


def _refined_band_db_frames(
    flac_path: Path,
    *,
    node: int,
    sample_ms: int,
    frames_per_block: int = 512,
) -> pd.DataFrame:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("soundfile is required to build v3 refined audio bands from FLAC") from exc

    start_time = _parse_flac_start(flac_path, node)
    info = sf.info(flac_path)
    frame_len = int(round(info.samplerate * sample_ms / 1000.0))
    if frame_len <= 0:
        raise ValueError("frame_len must be positive")

    window = np.hanning(frame_len).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / info.samplerate).astype(np.float32)
    rows: list[pd.DataFrame] = []
    frame_offset = 0
    blocksize = frame_len * frames_per_block

    for block in sf.blocks(flac_path, blocksize=blocksize, dtype="float32", always_2d=True):
        mono = _to_respeaker_mono(block)
        n_frames = len(mono) // frame_len
        if n_frames <= 0:
            continue

        frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
        centered = frames - frames.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(centered * window, axis=1)
        power = (np.abs(spectrum).astype(np.float32) ** 2).astype(np.float32)

        data: dict[str, np.ndarray] = {}
        for low_hz, high_hz in V3_REFINED_BANDS:
            band = _band_power(power, freqs, low_hz, high_hz)
            data[f"band_{low_hz}_{high_hz}_db"] = (10.0 * np.log10(band + FEATURE_EPS)).astype(np.float32)

        times = pd.date_range(
            start=start_time + pd.Timedelta(milliseconds=sample_ms * frame_offset),
            periods=n_frames,
            freq=f"{sample_ms}ms",
        )
        piece = pd.DataFrame(data)
        piece.insert(0, "datetime", times)
        piece.insert(1, "node", int(node))
        rows.append(piece)
        frame_offset += n_frames

    if not rows:
        raise RuntimeError(f"No frames extracted from {flac_path}")
    return pd.concat(rows, ignore_index=True)


def _load_or_build_v3_refined_audio_db(
    *,
    interim_dir: Path,
    raw_data_dir: Path,
    ordered_nodes: list[int],
    sample_ms: int,
) -> pd.DataFrame:
    cache_path = interim_dir / "audio_feature_v3_refined_db_long.csv"
    required = {"datetime", "node", *V3_REFINED_BAND_DB_FEATURES}
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["datetime"])
        if required.issubset(cached.columns):
            cached["node"] = cached["node"].astype(int)
            print(f"Loaded v3 refined audio-band cache: {cache_path} {cached.shape}")
            return cached

    pieces: list[pd.DataFrame] = []
    for node in ordered_nodes:
        flacs = sorted(raw_data_dir.glob(f"*_{node}_respeaker.flac"))
        if not flacs:
            raise FileNotFoundError(f"Missing FLAC for node {node} in {raw_data_dir}")
        for flac_path in flacs:
            node_df = _refined_band_db_frames(flac_path, node=node, sample_ms=sample_ms)
            pieces.append(node_df)
            print(f"v3 refined bands: node {node}, rows={len(node_df)}, file={flac_path.name}")

    refined = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["datetime", "node"])
        .drop_duplicates(["datetime", "node"], keep="first")
        .reset_index(drop=True)
    )
    refined.to_csv(cache_path, index=False)
    print(f"Saved v3 refined audio-band cache: {cache_path} {refined.shape}")
    return refined


def _require(paths: dict[str, Path], message: str) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(message + "\n" + "\n".join(missing))


def build_eq5_rssi_static_examples(
    project_root: str | Path | None = None,
    *,
    interim_name: str = DEFAULT_INTERIM_NAME,
    prefix: str = DEFAULT_PREFIX,
    exact_subset_size: int = 5,
    subset_size_policy: str = "exact",
    context_feature_mode: str = "rssi",
    audio_band_features: list[str] | None = None,
    utility_second_weight: float = 0.45,
    utility_third_weight: float = 0.20,
    rho: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build RSSI/static-action examples from interim observables.

    The resulting artifact intentionally uses no acoustic band features. Context
    contains normalized sensor coordinates plus the requested online sensor
    features; action vectors are static subset masks plus canonical slot
    coordinates. Set subset_size_policy="leq" to include all subset sizes
    from 1 to k.
    """

    root = resolve_project_root(project_root)
    interim_dir = root / "data" / "interim" / interim_name
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if context_feature_mode not in {"rssi", "feature_bands", "rssi_plus_feature_bands"}:
        raise ValueError(
            'context_feature_mode must be "rssi", "feature_bands", or "rssi_plus_feature_bands"'
        )
    if audio_band_features is None:
        audio_band_features = list(V3_REFINED_BAND_DB_FEATURES)

    required_paths = {
        "metadata": interim_dir / "metadata.json",
        "time_index": interim_dir / "time_index.csv",
        "rssi_db": interim_dir / "rssi_db.csv",
        "vehicle_node_distances": interim_dir / "vehicle_node_distances.csv",
        "sensor_geometry": interim_dir / "sensor_geometry.csv",
    }
    if context_feature_mode in {"feature_bands", "rssi_plus_feature_bands"}:
        required_paths["audio_feature_db_long"] = interim_dir / "audio_feature_db_long.csv"
    _require(required_paths, "Run the m01 interim export cell first. Missing interim files:")

    started = time.time()
    with open(required_paths["metadata"], "r", encoding="utf-8") as f:
        interim_meta = json.load(f)

    time_index = pd.read_csv(required_paths["time_index"], parse_dates=["datetime"])
    rssi_db = pd.read_csv(required_paths["rssi_db"], parse_dates=["datetime"])
    dist_long = pd.read_csv(required_paths["vehicle_node_distances"], parse_dates=["datetime"])
    sensor_geometry = pd.read_csv(required_paths["sensor_geometry"])

    ordered_nodes = [
        int(node)
        for node in interim_meta.get("nodes", sorted(sensor_geometry["node"].astype(int).tolist()))
    ]
    vehicle_labels = list(
        interim_meta.get("vehicles", sorted(dist_long["vehicle"].astype(str).unique().tolist()))
    )
    n_nodes = len(ordered_nodes)
    n_vehicles = len(vehicle_labels)
    if exact_subset_size > n_nodes:
        raise ValueError(f"exact_subset_size={exact_subset_size} exceeds n_nodes={n_nodes}")
    if subset_size_policy not in {"exact", "leq"}:
        raise ValueError('subset_size_policy must be either "exact" or "leq"')

    valid_times = (
        time_index[time_index["is_valid_model_timestep"].astype(int).eq(1)]
        .copy()
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    valid_times["time_id"] = np.arange(len(valid_times), dtype=np.int32)
    n_times = len(valid_times)

    subset_policy_label = (
        f"exact subset size={exact_subset_size}"
        if subset_size_policy == "exact"
        else f"subset sizes <= {exact_subset_size}"
    )
    print(
        f"Interim source: {interim_dir}\n"
        f"valid timestamps={n_times}, nodes={n_nodes}, vehicles={vehicle_labels}, "
        f"{subset_policy_label}"
    )

    rssi_cols = [f"rpi{node}" for node in ordered_nodes]
    missing_rssi = [col for col in rssi_cols if col not in rssi_db.columns]
    if missing_rssi:
        raise RuntimeError(f"Missing RSSI columns in rssi_db.csv: {missing_rssi}")

    rssi_valid = valid_times[["datetime", "time_id"]].merge(
        rssi_db[["datetime", *rssi_cols]],
        on="datetime",
        how="left",
    ).sort_values("time_id")
    rssi_values = rssi_valid[rssi_cols].astype(float)
    rssi_values = rssi_values.fillna(rssi_values.median()).fillna(rssi_values.stack().median()).fillna(0.0)
    rssi_mat = rssi_values.to_numpy(dtype=np.float32)

    sensor_geometry = sensor_geometry.copy()
    sensor_geometry["node"] = sensor_geometry["node"].astype(int)
    sensor_geometry = sensor_geometry.set_index("node").loc[ordered_nodes].reset_index()
    x_norm = sensor_geometry["sensor_x_norm"].to_numpy(dtype=np.float32)
    y_norm = sensor_geometry["sensor_y_norm"].to_numpy(dtype=np.float32)
    node_to_idx = {node: i for i, node in enumerate(ordered_nodes)}
    node_xy = {node: (float(x_norm[i]), float(y_norm[i])) for i, node in enumerate(ordered_nodes)}

    audio_feature_mats: dict[str, np.ndarray] = {}
    if context_feature_mode in {"feature_bands", "rssi_plus_feature_bands"}:
        audio_long = pd.read_csv(required_paths["audio_feature_db_long"], parse_dates=["datetime"])
        missing_audio = [feature for feature in audio_band_features if feature not in audio_long.columns]
        if missing_audio and set(missing_audio).issubset(set(V3_REFINED_BAND_DB_FEATURES)):
            refined = _load_or_build_v3_refined_audio_db(
                interim_dir=interim_dir,
                raw_data_dir=Path(interim_meta["raw_data_dir"]),
                ordered_nodes=ordered_nodes,
                sample_ms=int(interim_meta.get("sample_ms", 200)),
            )
            audio_long = audio_long.merge(
                refined[["datetime", "node", *missing_audio]],
                on=["datetime", "node"],
                how="left",
            )
            missing_audio = [feature for feature in audio_band_features if feature not in audio_long.columns]
        if missing_audio:
            raise RuntimeError(f"Missing audio band feature columns: {missing_audio}")
        audio_valid = audio_long[audio_long["is_valid_model_timestep"].astype(int).eq(1)].copy()
        audio_valid["time_id"] = audio_valid["time_id"].astype(int)
        audio_valid["node"] = audio_valid["node"].astype(int)
        for feature in audio_band_features:
            pivot = (
                audio_valid
                .pivot_table(index="time_id", columns="node", values=feature, aggfunc="first")
                .reindex(index=np.arange(n_times), columns=ordered_nodes)
            )
            values = pivot.astype(float)
            values = values.fillna(values.median()).fillna(values.stack().median()).fillna(0.0)
            audio_feature_mats[feature] = values.to_numpy(dtype=np.float32)

    context_feature_names: list[str] = []
    context_parts: list[np.ndarray] = []
    for i, node in enumerate(ordered_nodes):
        context_feature_names.extend([f"n{node}_sensor_x_norm", f"n{node}_sensor_y_norm"])
        context_parts.append(np.full((n_times, 1), x_norm[i], dtype=np.float32))
        context_parts.append(np.full((n_times, 1), y_norm[i], dtype=np.float32))
    if context_feature_mode in {"rssi", "rssi_plus_feature_bands"}:
        for i, node in enumerate(ordered_nodes):
            context_feature_names.append(f"n{node}_rssi_db")
            context_parts.append(rssi_mat[:, i : i + 1].astype(np.float32))
    if context_feature_mode in {"feature_bands", "rssi_plus_feature_bands"}:
        for i, node in enumerate(ordered_nodes):
            for feature in audio_band_features:
                context_feature_names.append(f"n{node}_{feature}")
                context_parts.append(audio_feature_mats[feature][:, i : i + 1].astype(np.float32))
    c_by_time = np.hstack(context_parts).astype(np.float32)

    dist_valid = dist_long[dist_long["is_valid_model_timestep"].astype(int).eq(1)].copy()
    dist_valid["time_id"] = dist_valid["time_id"].astype(int)
    dist_valid["node"] = dist_valid["node"].astype(int)
    dist_valid["vehicle"] = dist_valid["vehicle"].astype(str)

    dist = np.empty((n_vehicles, n_times, n_nodes), dtype=np.float32)
    for v, vehicle in enumerate(vehicle_labels):
        pivot = (
            dist_valid[dist_valid["vehicle"].eq(vehicle)]
            .pivot_table(index="time_id", columns="node", values="distance_to_vehicle_m", aggfunc="first")
            .reindex(index=np.arange(n_times), columns=ordered_nodes)
        )
        if pivot.isna().any().any():
            missing_count = int(pivot.isna().sum().sum())
            raise RuntimeError(f"Missing {missing_count} distance entries for {vehicle}.")
        dist[v] = pivot.to_numpy(dtype=np.float32)

    closest_idx = np.argmin(dist, axis=2).astype(np.int16)
    closest_node = np.take(np.asarray(ordered_nodes, dtype=np.int16), closest_idx)
    rank = np.empty((n_vehicles, n_times, n_nodes), dtype=np.int16)
    for v in range(n_vehicles):
        order = np.argsort(dist[v], axis=1)
        row_idx = np.arange(n_times)[:, None]
        rank[v][row_idx, order] = np.arange(1, n_nodes + 1, dtype=np.int16)[None, :]

    all_d1 = dist.min(axis=2).reshape(-1)
    rho_value = float(np.median(all_d1[np.isfinite(all_d1)])) if rho is None else float(rho)
    rho_value = max(rho_value, 1e-6)
    print(f"rho={rho_value:.3f} m")

    if subset_size_policy == "exact":
        subsets = [tuple(c) for c in combinations(ordered_nodes, exact_subset_size)]
    else:
        subsets = [
            tuple(c)
            for subset_size in range(1, exact_subset_size + 1)
            for c in combinations(ordered_nodes, subset_size)
        ]
    n_actions = len(subsets)
    subset_key = np.asarray(["-".join(str(n) for n in subset) for subset in subsets], dtype=object)
    subset_sizes = np.asarray([len(subset) for subset in subsets], dtype=np.int16)
    subset_node_idxs = [
        np.asarray([node_to_idx[node] for node in subset], dtype=np.int16)
        for subset in subsets
    ]

    action_feature_names = [f"mask_n{node}" for node in ordered_nodes]
    for slot in range(1, exact_subset_size + 1):
        action_feature_names.extend([f"slot{slot}_sensor_x_norm", f"slot{slot}_sensor_y_norm"])
    action_feature_names.append("subset_size")

    a_catalog = np.zeros((n_actions, len(action_feature_names)), dtype=np.float32)
    for m, subset in enumerate(subsets):
        idxs = subset_node_idxs[m]
        a_catalog[m, idxs] = 1.0
        cursor = n_nodes
        for slot_idx, node in enumerate(subset):
            a_catalog[m, cursor + 2 * slot_idx] = node_xy[node][0]
            a_catalog[m, cursor + 2 * slot_idx + 1] = node_xy[node][1]
        a_catalog[m, -1] = len(subset)

    utility_full_by_vehicle = np.empty((n_vehicles, n_times, n_actions), dtype=np.float32)
    utility_d1_by_vehicle = np.empty((n_vehicles, n_times, n_actions), dtype=np.float32)
    d1_by_vehicle = np.empty((n_vehicles, n_times, n_actions), dtype=np.float32)
    d2_by_vehicle = np.empty((n_vehicles, n_times, n_actions), dtype=np.float32)
    d3_by_vehicle = np.empty((n_vehicles, n_times, n_actions), dtype=np.float32)
    d_rank_by_vehicle = np.full(
        (n_vehicles, n_times, n_actions, exact_subset_size),
        np.nan,
        dtype=np.float32,
    )
    contains_by_vehicle = np.zeros((n_vehicles, n_times, n_actions), dtype=np.int8)
    best_rank_by_vehicle = np.empty((n_vehicles, n_times, n_actions), dtype=np.int16)

    for m, idxs in enumerate(subset_node_idxs):
        for v in range(n_vehicles):
            selected_dist = dist[v][:, idxs]
            d_sorted = np.sort(selected_dist, axis=1)
            for rank_idx in range(min(selected_dist.shape[1], exact_subset_size)):
                d_rank_by_vehicle[v, :, m, rank_idx] = d_sorted[:, rank_idx].astype(np.float32)
            d1 = d_sorted[:, 0]
            d2 = d_sorted[:, 1] if selected_dist.shape[1] >= 2 else d1
            d3 = d_sorted[:, 2] if selected_dist.shape[1] >= 3 else d2
            d1_term = 1.0 / (1.0 + d1 / rho_value)
            d2_term = utility_second_weight / (1.0 + d2 / rho_value)
            d3_term = utility_third_weight / (1.0 + d3 / rho_value)
            utility_d1_by_vehicle[v, :, m] = d1_term.astype(np.float32)
            utility_full_by_vehicle[v, :, m] = (d1_term + d2_term + d3_term).astype(np.float32)
            d1_by_vehicle[v, :, m] = d1.astype(np.float32)
            d2_by_vehicle[v, :, m] = d2.astype(np.float32)
            d3_by_vehicle[v, :, m] = d3.astype(np.float32)
            contains_by_vehicle[v, :, m] = np.isin(closest_idx[v], idxs).astype(np.int8)
            best_rank_by_vehicle[v, :, m] = np.min(rank[v][:, idxs], axis=1).astype(np.int16)

    utility_sum_full = utility_full_by_vehicle.sum(axis=0).astype(np.float32)
    utility_sum_d1_only = utility_d1_by_vehicle.sum(axis=0).astype(np.float32)
    utility_min_d1_only = utility_d1_by_vehicle.min(axis=0).astype(np.float32)
    utility_balanced_d1 = (
        0.5 * utility_d1_by_vehicle.mean(axis=0) + 0.5 * utility_min_d1_only
    ).astype(np.float32)
    utility_balanced_full = (
        0.5 * utility_full_by_vehicle.mean(axis=0) + 0.5 * utility_full_by_vehicle.min(axis=0)
    ).astype(np.float32)

    contains_count = contains_by_vehicle.sum(axis=0).astype(np.int8)
    contains_fraction = (contains_count / float(n_vehicles)).astype(np.float32)
    contains_all = (contains_count == n_vehicles).astype(np.int8)
    contains_any = (contains_count > 0).astype(np.int8)
    worst_rank = best_rank_by_vehicle.max(axis=0).astype(np.int16)
    mean_rank = best_rank_by_vehicle.mean(axis=0).astype(np.float32)

    time_id = np.repeat(np.arange(n_times, dtype=np.int32), n_actions)
    action_id = np.tile(np.arange(n_actions, dtype=np.int32), n_times)
    examples_index = pd.DataFrame(
        {
            "time_id": time_id,
            "datetime": np.repeat(valid_times["datetime"].to_numpy(dtype="datetime64[ns]"), n_actions),
            "action_id": action_id,
            "subset_str": np.tile(subset_key, n_times),
            "subset_size": np.tile(subset_sizes, n_times),
            "utility": utility_balanced_d1.reshape(-1),
            "utility_sum_full": utility_sum_full.reshape(-1),
            "utility_sum_d1_only": utility_sum_d1_only.reshape(-1),
            "utility_min_d1_only": utility_min_d1_only.reshape(-1),
            "utility_balanced_d1": utility_balanced_d1.reshape(-1),
            "utility_balanced_full": utility_balanced_full.reshape(-1),
            "contains_closest_count": contains_count.reshape(-1),
            "contains_closest_fraction": contains_fraction.reshape(-1),
            "contains_closest_any": contains_any.reshape(-1),
            "contains_closest_all": contains_all.reshape(-1),
            "contains_closest_node": contains_all.reshape(-1),
            "worst_vehicle_rank_in_subset": worst_rank.reshape(-1),
            "mean_vehicle_rank_in_subset": mean_rank.reshape(-1),
            "d1_worst_vehicle": d1_by_vehicle.max(axis=0).reshape(-1),
            "d1_mean_vehicle": d1_by_vehicle.mean(axis=0).reshape(-1),
        }
    )
    for v, vehicle in enumerate(vehicle_labels):
        examples_index[f"utility_full_{vehicle}"] = utility_full_by_vehicle[v].reshape(-1)
        examples_index[f"utility_d1_{vehicle}"] = utility_d1_by_vehicle[v].reshape(-1)
        examples_index[f"d1_{vehicle}"] = d1_by_vehicle[v].reshape(-1)
        examples_index[f"d2_{vehicle}"] = d2_by_vehicle[v].reshape(-1)
        examples_index[f"d3_{vehicle}"] = d3_by_vehicle[v].reshape(-1)
        for rank_idx in range(3, exact_subset_size):
            examples_index[f"d{rank_idx + 1}_{vehicle}"] = d_rank_by_vehicle[v, :, :, rank_idx].reshape(-1)
        examples_index[f"contains_closest_node_{vehicle}"] = contains_by_vehicle[v].reshape(-1)
        examples_index[f"best_rank_in_subset_{vehicle}"] = best_rank_by_vehicle[v].reshape(-1)
        examples_index[f"closest_node_{vehicle}"] = np.repeat(closest_node[v], n_actions)

    feature_wide_df = pd.DataFrame(c_by_time, columns=context_feature_names)
    feature_wide_df.insert(0, "datetime", valid_times["datetime"].to_numpy(dtype="datetime64[ns]"))

    paths = {
        "arrays_npz": processed_dir / f"{prefix}_arrays.npz",
        "examples_index_csv": processed_dir / f"{prefix}_examples_index.csv",
        "feature_wide_csv": processed_dir / f"{prefix}_feature_wide.csv",
        "sensor_geometry_csv": processed_dir / f"{prefix}_sensor_geometry.csv",
        "meta_json": processed_dir / f"{prefix}_meta.json",
    }
    np.savez_compressed(
        paths["arrays_npz"],
        C_by_time=c_by_time,
        A_catalog=a_catalog,
        action_id=action_id,
        subset_key=subset_key,
        y_examples=examples_index["utility"].to_numpy(dtype=np.float32),
        example_time_id=time_id,
        example_subset_size=examples_index["subset_size"].to_numpy(dtype=np.int16),
        example_contains_closest=examples_index["contains_closest_node"].to_numpy(dtype=np.int8),
        example_contains_closest_fraction=examples_index["contains_closest_fraction"].to_numpy(dtype=np.float32),
        sequence_times=valid_times["datetime"].to_numpy(dtype="datetime64[ns]"),
    )
    examples_index.to_csv(paths["examples_index_csv"], index=False)
    feature_wide_df.to_csv(paths["feature_wide_csv"], index=False)
    sensor_geometry.to_csv(paths["sensor_geometry_csv"], index=False)

    meta = {
        "prefix": prefix,
        "source_interim_dir": str(interim_dir),
        "ordered_nodes": ordered_nodes,
        "vehicle_labels": vehicle_labels,
        "multi_vehicle": True,
        "num_vehicles": n_vehicles,
        "subset_size_policy": (
            f"exactly {exact_subset_size}" if subset_size_policy == "exact" else f"<= {exact_subset_size}"
        ),
        "exact_subset_size": exact_subset_size if subset_size_policy == "exact" else None,
        "max_subset_size": exact_subset_size,
        "num_times": int(n_times),
        "num_actions_per_time": int(n_actions),
        "num_examples": int(len(examples_index)),
        "context_feature_mode": context_feature_mode,
        "audio_band_features": list(audio_band_features),
        "rho": rho_value,
        "utility_second_weight": utility_second_weight,
        "utility_third_weight": utility_third_weight,
        "default_utility_column": "utility_balanced_d1",
        "utility_columns": [candidate["column"] for candidate in ALL_UTILITY_CANDIDATES],
        "context_dim": int(c_by_time.shape[1]),
        "action_raw_dim": int(a_catalog.shape[1]),
        "context_feature_names": context_feature_names,
        "action_feature_names": action_feature_names,
        "static_action_feature_names": action_feature_names,
        "contains_closest_fraction_definition": (
            "1 if both vehicles' closest sensors are selected, 0.5 if exactly one is selected, "
            "0 if none are selected."
        ),
        "leakage_note": (
            "Context uses only sensor coordinates and online sensor features selected by "
            "context_feature_mode. Vehicle positions/distances are used only for utilities "
            "and evaluation labels."
        ),
    }
    with open(paths["meta_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    summary = pd.DataFrame(
        [
            {
                "prefix": prefix,
                "times": n_times,
                "actions_per_time": n_actions,
                "examples": len(examples_index),
                "context_dim": c_by_time.shape[1],
                "action_dim": a_catalog.shape[1],
                "mean_contains_fraction_all_actions": float(examples_index["contains_closest_fraction"].mean()),
                "both_rate_all_actions": float(examples_index["contains_closest_all"].mean()),
                "elapsed_min": float((time.time() - started) / 60.0),
            }
        ]
    )
    path_df = pd.DataFrame([{"artifact": key, "path": str(value)} for key, value in paths.items()])
    print(f"Saved processed artifact: {prefix}")
    print(f"Elapsed: {summary.loc[0, 'elapsed_min']:.1f} min")
    return summary, path_df


class CatalogPairDataset(Dataset):
    def __init__(
        self,
        c_by_time: np.ndarray,
        a_catalog: np.ndarray,
        action_id: np.ndarray,
        y: np.ndarray,
        time_id: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.c_by_time = torch.from_numpy(c_by_time.astype(np.float32))
        self.a_catalog = torch.from_numpy(a_catalog.astype(np.float32))
        self.action_id = torch.from_numpy(action_id.astype(np.int64))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.time_id = torch.from_numpy(time_id.astype(np.int64))
        self.indices = torch.from_numpy(indices.astype(np.int64))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        return self.c_by_time[self.time_id[idx]], self.a_catalog[self.action_id[idx]], self.y[idx]


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        if name not in seen:
            out.append(candidate)
            seen.add(name)
    return out


def _softmin_weighted(values: np.ndarray, beta: float) -> np.ndarray:
    """Smooth approximation to the lower per-vehicle utility."""
    z = -float(beta) * values
    z = z - np.max(z, axis=0, keepdims=True)
    weights = np.exp(z)
    weights = weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
    return np.sum(weights * values, axis=0).astype(np.float32)


def _candidate_utility_matrix(
    examples_index: pd.DataFrame,
    meta: dict[str, Any],
    candidate: dict[str, Any],
    n_times: int,
    n_actions: int,
) -> np.ndarray:
    column = str(candidate["column"])
    if column in examples_index.columns:
        return examples_index[column].to_numpy(dtype=np.float32).reshape(n_times, n_actions)

    vehicle_labels = list(meta["vehicle_labels"])
    d1_utilities = np.stack(
        [
            examples_index[f"utility_d1_{vehicle}"]
            .to_numpy(dtype=np.float32)
            .reshape(n_times, n_actions)
            for vehicle in vehicle_labels
        ],
        axis=0,
    )
    kind = candidate.get("kind")

    if kind == "softmin_d1":
        return _softmin_weighted(d1_utilities, float(candidate.get("beta", 10.0)))

    if kind == "sum_plus_softmin_d1":
        softmin = _softmin_weighted(d1_utilities, float(candidate.get("beta", 10.0)))
        return (d1_utilities.mean(axis=0) + float(candidate.get("lam", 1.0)) * softmin).astype(np.float32)

    if kind == "harmonic_d1":
        eps = 1e-6
        return (
            len(vehicle_labels) / np.maximum(np.sum(1.0 / np.maximum(d1_utilities, eps), axis=0), eps)
        ).astype(np.float32)

    if kind == "sum_d1_minus_gap":
        gap = d1_utilities.max(axis=0) - d1_utilities.min(axis=0)
        return (d1_utilities.sum(axis=0) - float(candidate.get("lam", 0.25)) * gap).astype(np.float32)

    if kind == "top2_d1":
        rho = max(float(meta["rho"]), 1e-6)
        d2_utilities = np.stack(
            [
                (
                    1.0
                    / (
                        1.0
                        + examples_index[f"d2_{vehicle}"]
                        .to_numpy(dtype=np.float32)
                        .reshape(n_times, n_actions)
                        / rho
                    )
                ).astype(np.float32)
                for vehicle in vehicle_labels
            ],
            axis=0,
        )
        return (d1_utilities + float(candidate.get("alpha", 0.2)) * d2_utilities).sum(axis=0).astype(np.float32)

    if kind == "softmin_rank_weighted":
        rho = max(float(meta["rho"]), 1e-6)
        weights = np.asarray(candidate.get("weights", [1.0, 0.45, 0.20, 0.10, 0.05]), dtype=np.float32)
        per_vehicle_utilities: list[np.ndarray] = []
        for vehicle in vehicle_labels:
            total = np.zeros((n_times, n_actions), dtype=np.float32)
            for rank_idx, weight in enumerate(weights, start=1):
                if float(weight) == 0.0:
                    continue
                col = f"d{rank_idx}_{vehicle}"
                if col not in examples_index.columns:
                    raise RuntimeError(
                        f"Missing {col!r}; rebuild the processed artifact with rank distances up to d{len(weights)}."
                    )
                d = examples_index[col].to_numpy(dtype=np.float32).reshape(n_times, n_actions)
                term = np.zeros_like(d, dtype=np.float32)
                valid = np.isfinite(d)
                term[valid] = 1.0 / (1.0 + d[valid] / rho)
                total += float(weight) * term
            per_vehicle_utilities.append(total)
        return _softmin_weighted(np.stack(per_vehicle_utilities, axis=0), float(candidate.get("beta", 5.0)))

    raise RuntimeError(
        f"Cannot build utility candidate {candidate['name']!r}: "
        f"column {column!r} is missing and kind {kind!r} is unknown."
    )


def _build_or_load_dense_cache(
    processed_dir: Path,
    prefix: str,
    arrays: np.lib.npyio.NpzFile,
    meta: dict[str, Any],
    *,
    utility_candidates: list[dict[str, Any]] | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Load compact time x action arrays for dense training/evaluation.

    The CSV example index is convenient for inspection but far too heavy to use
    every epoch. This cache turns the important columns into dense matrices with
    shape (num_times, num_actions).
    """

    requested_candidates = utility_candidates or UTILITY_CANDIDATES
    cache_candidates = _dedupe_candidates([*ALL_UTILITY_CANDIDATES, *requested_candidates])
    requested_names = [candidate["name"] for candidate in requested_candidates]
    cache_names = [candidate["name"] for candidate in cache_candidates]
    cache_path = processed_dir / f"{prefix}_dense_train_cache.npz"
    vehicle_labels = list(meta["vehicle_labels"])

    if cache_path.exists() and not force_rebuild:
        cached = np.load(cache_path, allow_pickle=True)
        missing_keys = [f"utility__{name}" for name in requested_names if f"utility__{name}" not in cached.files]
        if not missing_keys:
            return {
                "cache_path": cache_path,
                "utility": {name: cached[f"utility__{name}"].astype(np.float32) for name in requested_names},
                "contains_fraction": cached["contains_fraction"].astype(np.float32),
                "contains_count": cached["contains_count"].astype(np.int16),
                "contains_all": cached["contains_all"].astype(np.float32),
                "contains_any": cached["contains_any"].astype(np.float32),
                "worst_rank": cached["worst_rank"].astype(np.float32),
                "mean_rank": cached["mean_rank"].astype(np.float32),
                "vehicle_contains": {
                    vehicle: cached[f"contains__{vehicle}"].astype(np.float32)
                    for vehicle in vehicle_labels
                },
                "vehicle_rank": {
                    vehicle: cached[f"rank__{vehicle}"].astype(np.float32)
                    for vehicle in vehicle_labels
                },
                "vehicle_d1": {
                    vehicle: cached[f"d1__{vehicle}"].astype(np.float32)
                    for vehicle in vehicle_labels
                },
            }
        print(
            "Dense cache is missing requested utility targets; rebuilding with "
            f"{len(cache_names)} utility matrices."
        )
        cached.close()

    examples_path = processed_dir / f"{prefix}_examples_index.csv"
    if not examples_path.exists():
        raise RuntimeError(f"Missing examples index CSV needed to build dense cache: {examples_path}")

    print(f"Building dense train cache from {examples_path.name} ...")
    examples_index = pd.read_csv(examples_path)
    n_times = int(arrays["C_by_time"].shape[0])
    n_actions = int(arrays["A_catalog"].shape[0])
    examples_index = examples_index.sort_values(["time_id", "action_id"]).reset_index(drop=True)

    expected_time = np.repeat(np.arange(n_times, dtype=np.int32), n_actions)
    expected_action = np.tile(np.arange(n_actions, dtype=np.int32), n_times)
    if not np.array_equal(examples_index["time_id"].to_numpy(dtype=np.int32), expected_time):
        raise RuntimeError("examples_index time_id order is not dense time-major order.")
    if not np.array_equal(examples_index["action_id"].to_numpy(dtype=np.int32), expected_action):
        raise RuntimeError("examples_index action_id order is not dense action-major within each time.")

    save_kwargs: dict[str, np.ndarray] = {
        "contains_fraction": examples_index["contains_closest_fraction"].to_numpy(dtype=np.float32).reshape(n_times, n_actions),
        "contains_count": examples_index["contains_closest_count"].to_numpy(dtype=np.int16).reshape(n_times, n_actions),
        "contains_all": examples_index["contains_closest_all"].to_numpy(dtype=np.float32).reshape(n_times, n_actions),
        "contains_any": examples_index["contains_closest_any"].to_numpy(dtype=np.float32).reshape(n_times, n_actions),
        "worst_rank": examples_index["worst_vehicle_rank_in_subset"].to_numpy(dtype=np.float32).reshape(n_times, n_actions),
        "mean_rank": examples_index["mean_vehicle_rank_in_subset"].to_numpy(dtype=np.float32).reshape(n_times, n_actions),
    }
    for candidate in cache_candidates:
        save_kwargs[f"utility__{candidate['name']}"] = _candidate_utility_matrix(
            examples_index,
            meta,
            candidate,
            n_times,
            n_actions,
        )
    for vehicle in vehicle_labels:
        save_kwargs[f"contains__{vehicle}"] = (
            examples_index[f"contains_closest_node_{vehicle}"]
            .to_numpy(dtype=np.float32)
            .reshape(n_times, n_actions)
        )
        save_kwargs[f"rank__{vehicle}"] = (
            examples_index[f"best_rank_in_subset_{vehicle}"]
            .to_numpy(dtype=np.float32)
            .reshape(n_times, n_actions)
        )
        save_kwargs[f"d1__{vehicle}"] = (
            examples_index[f"d1_{vehicle}"]
            .to_numpy(dtype=np.float32)
            .reshape(n_times, n_actions)
        )

    np.savez_compressed(cache_path, **save_kwargs)
    print(f"Saved dense train cache: {cache_path}")
    cached = np.load(cache_path, allow_pickle=True)
    return {
        "cache_path": cache_path,
        "utility": {name: cached[f"utility__{name}"].astype(np.float32) for name in requested_names},
        "contains_fraction": cached["contains_fraction"].astype(np.float32),
        "contains_count": cached["contains_count"].astype(np.int16),
        "contains_all": cached["contains_all"].astype(np.float32),
        "contains_any": cached["contains_any"].astype(np.float32),
        "worst_rank": cached["worst_rank"].astype(np.float32),
        "mean_rank": cached["mean_rank"].astype(np.float32),
        "vehicle_contains": {
            vehicle: cached[f"contains__{vehicle}"].astype(np.float32)
            for vehicle in vehicle_labels
        },
        "vehicle_rank": {
            vehicle: cached[f"rank__{vehicle}"].astype(np.float32)
            for vehicle in vehicle_labels
        },
        "vehicle_d1": {
            vehicle: cached[f"d1__{vehicle}"].astype(np.float32)
            for vehicle in vehicle_labels
        },
    }


def chronological_split(
    example_time_id: np.ndarray,
    n_times: int,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> dict[str, np.ndarray]:
    n_train = int(train_frac * n_times)
    n_val = int(val_frac * n_times)
    train_time_ids = np.arange(0, n_train)
    val_time_ids = np.arange(n_train, n_train + n_val)
    test_time_ids = np.arange(n_train + n_val, n_times)
    return {
        "train": np.flatnonzero(np.isin(example_time_id, train_time_ids)),
        "val": np.flatnonzero(np.isin(example_time_id, val_time_ids)),
        "test": np.flatnonzero(np.isin(example_time_id, test_time_ids)),
        "train_time_ids": train_time_ids,
        "val_time_ids": val_time_ids,
        "test_time_ids": test_time_ids,
    }


def _score_all_actions(
    model: torch.nn.Module,
    context_batch: torch.Tensor,
    action_catalog: torch.Tensor,
) -> torch.Tensor:
    """Score a batch of timestamps against every static action."""

    c_emb = model.embed_context(context_batch)
    a_emb = model.embed_action(action_catalog)
    bsz, emb_dim = c_emb.shape
    n_actions = a_emb.shape[0]
    c_pair = c_emb[:, None, :].expand(bsz, n_actions, emb_dim).reshape(-1, emb_dim)
    a_pair = a_emb[None, :, :].expand(bsz, n_actions, emb_dim).reshape(-1, emb_dim)
    return model.score_embeddings(c_pair, a_pair).reshape(bsz, n_actions)


@torch.no_grad()
def _predict_dense_scores(
    model: torch.nn.Module,
    c_std: np.ndarray,
    a_catalog_std: np.ndarray,
    time_ids: np.ndarray,
    time_batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    action_tensor = torch.from_numpy(a_catalog_std.astype(np.float32)).to(device)
    outputs = []
    for start in range(0, len(time_ids), time_batch_size):
        tids = time_ids[start : start + time_batch_size]
        c_batch = torch.from_numpy(c_std[tids].astype(np.float32)).to(device)
        outputs.append(_score_all_actions(model, c_batch, action_tensor).detach().cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def _dense_regression_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    err = scores - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return {"rmse": rmse, "mae": mae, "r2": float(r2)}


def _evaluate_dense(
    scores: np.ndarray,
    y_true: np.ndarray,
    cache: dict[str, Any],
    vehicle_labels: list[str],
    time_ids: np.ndarray,
) -> dict[str, float]:
    selected_action = np.argmax(scores, axis=1)
    selected_y = y_true[np.arange(len(time_ids)), selected_action]
    best_y = y_true.max(axis=1)
    worst_y = y_true.min(axis=1)
    rank = (y_true > selected_y[:, None] + 1e-10).sum(axis=1) + 1
    regret = best_y - selected_y
    denom = best_y - worst_y
    norm_regret = np.where(denom <= 1e-12, 0.0, regret / denom)

    global_rows = (time_ids, selected_action)
    out = _dense_regression_metrics(y_true, scores)
    out.update(
        {
            "top1": float(np.mean(rank == 1)),
            "top3": float(np.mean(rank <= 3)),
            "mean_rank": float(np.mean(rank)),
            "avg_regret": float(np.mean(regret)),
            "avg_norm_regret": float(np.mean(norm_regret)),
            "n_timestamps": int(len(time_ids)),
            "contains_closest_fraction": float(np.mean(cache["contains_fraction"][global_rows])),
            "contains_closest_all": float(np.mean(cache["contains_all"][global_rows])),
            "contains_closest_any": float(np.mean(cache["contains_any"][global_rows])),
            "neither_closest_rate": float(np.mean(cache["contains_count"][global_rows] == 0)),
            "one_closest_rate": float(np.mean(cache["contains_count"][global_rows] == 1)),
            "both_closest_rate": float(np.mean(cache["contains_count"][global_rows] == len(vehicle_labels))),
            "mean_worst_vehicle_rank": float(np.mean(cache["worst_rank"][global_rows])),
            "mean_vehicle_rank": float(np.mean(cache["mean_rank"][global_rows])),
        }
    )

    vehicle_contains = {}
    for vehicle in vehicle_labels:
        contains = cache["vehicle_contains"][vehicle][global_rows]
        ranks = cache["vehicle_rank"][vehicle][global_rows]
        d1 = cache["vehicle_d1"][vehicle][global_rows]
        vehicle_contains[vehicle] = float(np.mean(contains))
        out[f"contains_{vehicle}"] = vehicle_contains[vehicle]
        out[f"mean_rank_{vehicle}"] = float(np.mean(ranks))
        out[f"mean_d1_{vehicle}"] = float(np.mean(d1))

    if len(vehicle_labels) == 2:
        v1, v2 = vehicle_labels
        c1 = cache["vehicle_contains"][v1][global_rows].astype(int)
        c2 = cache["vehicle_contains"][v2][global_rows].astype(int)
        r1 = cache["vehicle_rank"][v1][global_rows].astype(float)
        r2 = cache["vehicle_rank"][v2][global_rows].astype(float)
        d1 = cache["vehicle_d1"][v1][global_rows].astype(float)
        d2 = cache["vehicle_d1"][v2][global_rows].astype(float)
        out.update(
            {
                "dominance_gap": float(abs(vehicle_contains[v1] - vehicle_contains[v2])),
                f"{v1}_only_rate": float(np.mean((c1 == 1) & (c2 == 0))),
                f"{v2}_only_rate": float(np.mean((c1 == 0) & (c2 == 1))),
                "mean_abs_vehicle_rank_gap": float(np.mean(np.abs(r1 - r2))),
                "mean_abs_d1_gap_m": float(np.mean(np.abs(d1 - d2))),
            }
        )
    return out


def _make_loader(
    c_std: np.ndarray,
    a_catalog_std: np.ndarray,
    action_id: np.ndarray,
    y: np.ndarray,
    example_time_id: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = CatalogPairDataset(c_std, a_catalog_std, action_id, y, example_time_id, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _predict_scores(
    model: torch.nn.Module,
    c_std: np.ndarray,
    a_catalog_std: np.ndarray,
    action_id: np.ndarray,
    y: np.ndarray,
    example_time_id: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    loader = _make_loader(c_std, a_catalog_std, action_id, y, example_time_id, indices, batch_size, False)
    model.eval()
    outputs = []
    for c_batch, a_batch, _ in loader:
        c_batch = c_batch.to(device)
        a_batch = a_batch.to(device)
        outputs.append(model(c_batch, a_batch).detach().cpu().numpy())
    return np.concatenate(outputs)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return {"rmse": rmse, "mae": mae, "r2": float(r2)}


def _evaluate_indices(
    examples_index: pd.DataFrame,
    vehicle_labels: list[str],
    indices: np.ndarray,
    scores: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    eval_df = examples_index.iloc[indices].copy()
    eval_df["_score"] = scores.astype(float)
    eval_df["_y"] = y[indices].astype(float)
    selected_idx = eval_df.groupby("time_id", sort=False)["_score"].idxmax()
    selected = eval_df.loc[selected_idx].copy()

    top1_hits = []
    top3_hits = []
    ranks = []
    regrets = []
    norm_regrets = []
    for _, group in eval_df.groupby("time_id", sort=False):
        y_true = group["_y"].to_numpy(dtype=float)
        score = group["_score"].to_numpy(dtype=float)
        order = np.argsort(-score)
        selected_value = float(y_true[int(order[0])])
        best_value = float(np.max(y_true))
        worst_value = float(np.min(y_true))
        true_rank = int(np.sum(y_true > selected_value + 1e-10)) + 1
        top1_hits.append(float(true_rank == 1))
        top3_hits.append(float(true_rank <= 3))
        ranks.append(true_rank)
        regret = best_value - selected_value
        regrets.append(regret)
        denom = best_value - worst_value
        norm_regrets.append(0.0 if denom <= 1e-12 else regret / denom)

    out = _regression_metrics(y[indices], scores)
    out.update(
        {
            "top1": float(np.mean(top1_hits)),
            "top3": float(np.mean(top3_hits)),
            "mean_rank": float(np.mean(ranks)),
            "avg_regret": float(np.mean(regrets)),
            "avg_norm_regret": float(np.mean(norm_regrets)),
            "n_timestamps": int(selected["time_id"].nunique()),
            "contains_closest_fraction": float(selected["contains_closest_fraction"].mean()),
            "contains_closest_all": float(selected["contains_closest_all"].mean()),
            "contains_closest_any": float(selected["contains_closest_any"].mean()),
            "neither_closest_rate": float((selected["contains_closest_count"].to_numpy() == 0).mean()),
            "one_closest_rate": float((selected["contains_closest_count"].to_numpy() == 1).mean()),
            "both_closest_rate": float(
                (selected["contains_closest_count"].to_numpy() == len(vehicle_labels)).mean()
            ),
            "mean_worst_vehicle_rank": float(selected["worst_vehicle_rank_in_subset"].mean()),
            "mean_vehicle_rank": float(selected["mean_vehicle_rank_in_subset"].mean()),
        }
    )

    vehicle_contains: dict[str, float] = {}
    for vehicle in vehicle_labels:
        vehicle_contains[vehicle] = float(selected[f"contains_closest_node_{vehicle}"].mean())
        out[f"contains_{vehicle}"] = vehicle_contains[vehicle]
        out[f"mean_rank_{vehicle}"] = float(selected[f"best_rank_in_subset_{vehicle}"].mean())
        out[f"mean_d1_{vehicle}"] = float(selected[f"d1_{vehicle}"].mean())

    if len(vehicle_labels) == 2:
        v1, v2 = vehicle_labels
        c1 = selected[f"contains_closest_node_{v1}"].to_numpy(dtype=int)
        c2 = selected[f"contains_closest_node_{v2}"].to_numpy(dtype=int)
        r1 = selected[f"best_rank_in_subset_{v1}"].to_numpy(dtype=float)
        r2 = selected[f"best_rank_in_subset_{v2}"].to_numpy(dtype=float)
        d1 = selected[f"d1_{v1}"].to_numpy(dtype=float)
        d2 = selected[f"d1_{v2}"].to_numpy(dtype=float)
        out.update(
            {
                "dominance_gap": float(abs(vehicle_contains[v1] - vehicle_contains[v2])),
                f"{v1}_only_rate": float(((c1 == 1) & (c2 == 0)).mean()),
                f"{v2}_only_rate": float(((c1 == 0) & (c2 == 1)).mean()),
                "mean_abs_vehicle_rank_gap": float(np.mean(np.abs(r1 - r2))),
                "mean_abs_d1_gap_m": float(np.mean(np.abs(d1 - d2))),
            }
        )
    return out


def _selector_key(val: dict[str, float]) -> tuple[float, ...]:
    return (
        -float(val["contains_closest_fraction"]),
        -float(val["contains_closest_all"]),
        float(val.get("dominance_gap", 0.0)),
        float(val["mean_worst_vehicle_rank"]),
        float(val["avg_regret"]),
        float(val["rmse"]),
    )


def run_rssi_utility_search(
    project_root: str | Path | None = None,
    *,
    prefix: str = DEFAULT_PREFIX,
    model_tag: str = DEFAULT_MODEL_TAG,
    max_epochs: int = 100,
    patience: int = 35,
    log_every: int = 5,
    batch_size: int = 8192,
    time_batch_size: int = 256,
    seed: int = 22,
    device: str | None = None,
    utility_candidates: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    root = resolve_project_root(project_root)
    processed_dir = root / "data" / "processed"
    out_dir = root / "experiments" / model_tag
    table_dir = root / "reports" / "tables" / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    arrays_path = processed_dir / f"{prefix}_arrays.npz"
    meta_path = processed_dir / f"{prefix}_meta.json"
    _require(
        {"arrays": arrays_path, "meta": meta_path},
        "Run the build cell first. Missing processed files:",
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    utility_candidates = utility_candidates or UTILITY_CANDIDATES

    arrays = np.load(arrays_path, allow_pickle=True)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    c_by_time = arrays["C_by_time"].astype(np.float32)
    a_catalog = arrays["A_catalog"].astype(np.float32)
    vehicle_labels = list(meta["vehicle_labels"])
    n_times = c_by_time.shape[0]
    n_actions = a_catalog.shape[0]

    cache = _build_or_load_dense_cache(
        processed_dir,
        prefix,
        arrays,
        meta,
        utility_candidates=utility_candidates,
    )

    print("PROJECT_ROOT:", root)
    print("DEVICE:", device)
    print(
        f"data: times={n_times}, actions/time={n_actions}, "
        f"examples={n_times * n_actions:,}, context_dim={c_by_time.shape[1]}, "
        f"action_dim={a_catalog.shape[1]}"
    )
    print(f"dense cache: {cache['cache_path']}")
    print("utilities:", [candidate["name"] for candidate in utility_candidates])

    split = {
        "train_time_ids": np.arange(0, int(0.60 * n_times)),
        "val_time_ids": np.arange(int(0.60 * n_times), int(0.80 * n_times)),
        "test_time_ids": np.arange(int(0.80 * n_times), n_times),
    }
    c_mu, c_sigma = fit_standardizer(c_by_time[split["train_time_ids"]])
    a_mu, a_sigma = fit_standardizer(a_catalog)
    c_std = ((c_by_time - c_mu) / c_sigma).astype(np.float32)
    a_catalog_std = ((a_catalog - a_mu) / a_sigma).astype(np.float32)
    action_tensor = torch.from_numpy(a_catalog_std.astype(np.float32)).to(device)

    results: list[dict[str, Any]] = []
    for candidate in utility_candidates:
        y_matrix = cache["utility"][candidate["name"]]
        cfg = TrainConfig(
            run_name=f"mv_eq5_rssi_{candidate['name']}_h512_d2_e16",
            utility_name=candidate["name"],
            hidden=512,
            emb_dim=16,
            depth=2,
            dropout=0.05,
            combine_mode="mul_only",
            loss_name="mse",
            lr=5e-4,
            weight_decay=1e-4,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            train_frac=0.60,
            val_frac=0.20,
            log_every=log_every,
            num_workers=0,
        )

        set_all_seeds(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        model = TwoTowerMLP(
            context_dim=c_by_time.shape[1],
            action_dim=a_catalog.shape[1],
            hidden=cfg.hidden,
            emb_dim=cfg.emb_dim,
            depth=cfg.depth,
            dropout=cfg.dropout,
            combine_mode=cfg.combine_mode,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = make_loss(cfg.loss_name)

        best_state = None
        best_epoch = -1
        best_key = None
        wait = 0
        history_rows = []
        start = time.time()
        run_dir = out_dir / candidate["name"]
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {candidate['name']} === {candidate['description']}")
        for epoch in range(1, cfg.max_epochs + 1):
            model.train()
            losses = []
            train_time_ids = rng.permutation(split["train_time_ids"])
            for batch_start in range(0, len(train_time_ids), time_batch_size):
                tids = train_time_ids[batch_start : batch_start + time_batch_size]
                c_batch = torch.from_numpy(c_std[tids].astype(np.float32)).to(device)
                y_batch = torch.from_numpy(y_matrix[tids].astype(np.float32)).to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = _score_all_actions(model, c_batch, action_tensor)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(losses))
            val_scores = _predict_dense_scores(
                model,
                c_std,
                a_catalog_std,
                split["val_time_ids"],
                time_batch_size,
                device,
            )
            val = _evaluate_dense(
                val_scores,
                y_matrix[split["val_time_ids"]],
                cache,
                vehicle_labels,
                split["val_time_ids"],
            )
            key = _selector_key(val)
            improved = best_key is None or key < best_key
            if improved:
                best_key = key
                best_epoch = int(epoch)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            history_rows.append(
                {
                    "epoch": int(epoch),
                    "train_loss": train_loss,
                    "is_best_epoch_so_far": bool(improved),
                    **{f"val_{k}": float(v) for k, v in val.items()},
                }
            )
            if epoch == 1 or epoch % cfg.log_every == 0 or improved or epoch == cfg.max_epochs:
                star = "*" if improved else " "
                vehicle_rates = " ".join([f"{v}={val[f'contains_{v}']:.3f}" for v in vehicle_labels])
                print(
                    f"{candidate['name']:>16s} ep {epoch:03d}/{cfg.max_epochs:03d}{star} "
                    f"loss={train_loss:.5f} val_acc={val['contains_closest_fraction']:.3f} "
                    f"both={val['both_closest_rate']:.3f} one={val['one_closest_rate']:.3f} "
                    f"gap={val.get('dominance_gap', 0.0):.3f} {vehicle_rates} "
                    f"wrank={val['mean_worst_vehicle_rank']:.2f} reg={val['avg_regret']:.4f} "
                    f"{time.time() - start:.0f}s"
                )
            if wait >= cfg.patience:
                print(f"early stop at epoch {epoch}; selected epoch {best_epoch}")
                break

        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for {candidate['name']}")
        model.load_state_dict(best_state)

        metrics = {}
        metric_rows = []
        for split_name, time_key in (
            ("train", "train_time_ids"),
            ("val", "val_time_ids"),
            ("test", "test_time_ids"),
        ):
            time_ids = split[time_key]
            scores = _predict_dense_scores(
                model,
                c_std,
                a_catalog_std,
                time_ids,
                time_batch_size,
                device,
            )
            metrics[split_name] = _evaluate_dense(
                scores,
                y_matrix[time_ids],
                cache,
                vehicle_labels,
                time_ids,
            )
            metric_rows.append(
                {
                    "utility": candidate["name"],
                    "utility_column": candidate["column"],
                    "split": split_name,
                    "best_epoch": int(best_epoch),
                    **metrics[split_name],
                }
            )

        history_df = pd.DataFrame(history_rows)
        metrics_df = pd.DataFrame(metric_rows)
        history_df.to_csv(table_dir / f"{candidate['name']}_history.csv", index=False)
        metrics_df.to_csv(table_dir / f"{candidate['name']}_metrics.csv", index=False)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "candidate": candidate,
                "best_epoch": int(best_epoch),
                "metrics": metrics,
                "standardizers": {"C_mu": c_mu, "C_sigma": c_sigma, "A_mu": a_mu, "A_sigma": a_sigma},
                "meta": meta,
                "selector_rule": (
                    "maximize val contains_closest_fraction, then contains_all, then minimize "
                    "dominance_gap, worst-rank, regret, rmse"
                ),
            },
            run_dir / "best_model.pt",
        )
        results.append(
            {
                "candidate": candidate,
                "config": cfg,
                "best_epoch": best_epoch,
                "history": history_df,
                "metrics": metrics,
                "metrics_df": metrics_df,
            }
        )

    summary_rows = []
    for result in results:
        candidate = result["candidate"]
        row = {
            "utility": candidate["name"],
            "utility_column": candidate["column"],
            "best_epoch": int(result["best_epoch"]),
        }
        for split_name, metrics in result["metrics"].items():
            for key, value in metrics.items():
                row[f"{split_name}_{key}"] = value
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        [
            "test_contains_closest_fraction",
            "test_contains_closest_all",
            "test_dominance_gap",
            "test_mean_worst_vehicle_rank",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    summary_df.to_csv(table_dir / "utility_search_summary.csv", index=False)
    print(f"Saved histories/metrics to {table_dir}")
    return summary_df, results
