from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONTEXT_AUDIO_FEATURES = [
    "rms_db",
    "spectral_centroid_hz",
    "spectral_flatness",
    "band_20_120_ratio",
    "band_120_500_ratio",
    "band_500_2000_ratio",
    "low_to_voice_db",
    "mid_to_voice_db",
    "rms_delta_db",
    "centroid_delta_hz",
]

EXPANDED_CONTEXT_AUDIO_FEATURES = [
    "rms_db",
    "spectral_centroid_hz",
    "spectral_flatness",
    "band_20_120_ratio",
    "band_120_500_ratio",
    "band_500_2000_ratio",
    "low_to_voice_db",
    "mid_to_voice_db",
    "rms_delta_db",
    "centroid_delta_hz",
    "peak_db",
    "crest_factor_db",
    "zcr",
    "spectral_bandwidth_hz",
    "spectral_rolloff85_hz",
    "spectral_entropy",
    "band_20_120_db",
    "band_120_500_db",
    "band_500_2000_db",
    "band_2000_6000_db",
    "band_2000_6000_ratio",
    "flatness_delta",
]

DEFAULT_ACTION_AUDIO_FEATURES = [
    "rms_db",
    "spectral_centroid_hz",
    "spectral_flatness",
    "band_20_120_ratio",
    "band_120_500_ratio",
    "low_to_voice_db",
]

EXPANDED_ACTION_AUDIO_FEATURES = [
    "rms_db",
    "spectral_centroid_hz",
    "spectral_flatness",
    "band_20_120_ratio",
    "band_120_500_ratio",
    "low_to_voice_db",
    "band_500_2000_ratio",
    "mid_to_voice_db",
    "crest_factor_db",
    "zcr",
    "spectral_bandwidth_hz",
    "spectral_rolloff85_hz",
    "spectral_entropy",
    "band_2000_6000_ratio",
    "flatness_delta",
]

DERIVED_AUDIO_SCORE_FEATURES = [
    "vehicle_like_score",
    "construction_like_score",
    "vehicle_minus_construction_score",
]


def _linear_slope(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x), dtype=float)
    y = np.asarray(x, dtype=float)
    denom = np.sum((t - t.mean()) ** 2)
    if denom <= 0:
        return 0.0
    return float(np.sum((t - t.mean()) * (y - y.mean())) / denom)


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _iqr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def _trimmed_mean(x: np.ndarray, proportion: float = 0.10) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) < 3:
        return float(np.mean(x))
    k = int(np.floor(len(x) * proportion))
    if 2 * k >= len(x):
        return float(np.mean(x))
    return float(np.mean(x[k : len(x) - k]))


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(x).any() else 0.0)
    z = x / max(float(temperature), 1e-8)
    z = z - np.max(z)
    ex = np.exp(z)
    denom = np.sum(ex)
    if denom <= 0:
        return np.ones_like(x) / len(x)
    return ex / denom


def _normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    denom = s.max() - s.min()
    if denom <= 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / denom


def _node_col(node: int) -> str:
    return f"rpi{node}"


def _dist_col(node: int) -> str:
    return f"distance_to_{node}"


def _detect_vehicles(gdf_cleaned: pd.DataFrame) -> list[str]:
    """
    Detect vehicle labels in the dataframe by looking for geometry_vehicle* columns.
    Returns sorted list of vehicle labels like ['vehicle1', 'vehicle2', ...].
    """
    vehicle_labels = []
    for col in gdf_cleaned.columns:
        if col.startswith("geometry_vehicle"):
            label = col.replace("geometry_", "")
            vehicle_labels.append(label)
    return sorted(vehicle_labels, key=lambda x: int(x.replace("vehicle", "")))


def _get_distance_columns(gdf_cleaned: pd.DataFrame, node: int, vehicle_id: str | None = None) -> str:
    """
    Get the distance column name for a node, handling both single and multi-vehicle cases.
    
    If vehicle_id is specified, tries to get vehicle-specific distance.
    Otherwise, uses the first available distance column for the node.
    """
    if vehicle_id:
        col = f"distance_to_{node}_{vehicle_id}"
        if col in gdf_cleaned.columns:
            return col
    
    # Try single-vehicle naming first
    single_col = f"distance_to_{node}"
    if single_col in gdf_cleaned.columns:
        return single_col
    
    # Try multi-vehicle naming with first vehicle
    for col in sorted(gdf_cleaned.columns):
        if col.startswith(f"distance_to_{node}_vehicle"):
            return col
    
    raise ValueError(f"No distance column found for node {node}")


def _subset_to_str(subset: tuple[int, ...]) -> str:
    return "-".join(map(str, subset))


def _triangle_area(points: list[tuple[float, float]]) -> float:
    if len(points) != 3:
        return 0.0
    (x1, y1), (x2, y2), (x3, y3) = points
    return float(abs(0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))))


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _as_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _clean_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.isfinite(values).any():
        fill = float(np.nanmedian(values[np.isfinite(values)]))
    else:
        fill = 0.0
    return np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill)


def _robust_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    med = s.median(skipna=True)
    mad = np.median(np.abs(s.dropna().to_numpy(dtype=float) - med)) if s.notna().any() else 0.0
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-8:
        scale = s.std(skipna=True)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return ((s - med) / scale).fillna(0.0)


def _add_audio_signature_scores(audio_feature_long: pd.DataFrame | None) -> pd.DataFrame | None:
    if audio_feature_long is None:
        return None

    audio = audio_feature_long.copy()

    def z(name: str) -> pd.Series:
        if name in audio.columns:
            return _robust_z(audio[name])
        return pd.Series(np.zeros(len(audio), dtype=float), index=audio.index)

    vehicle_like = (
        0.85 * z("rms_db")
        + 0.80 * z("band_20_120_ratio")
        + 0.45 * z("band_120_500_ratio")
        - 0.35 * z("spectral_entropy")
        - 0.25 * z("spectral_flatness")
        - 0.20 * z("zcr")
    )
    construction_like = (
        0.70 * z("band_2000_6000_ratio")
        + 0.45 * z("spectral_entropy")
        + 0.35 * z("zcr")
        + 0.30 * z("crest_factor_db")
        + 0.20 * z("spectral_rolloff85_hz")
        - 0.25 * z("band_20_120_ratio")
    )

    audio["vehicle_like_score"] = vehicle_like.astype(float)
    audio["construction_like_score"] = construction_like.astype(float)
    audio["vehicle_minus_construction_score"] = (
        audio["vehicle_like_score"] - audio["construction_like_score"]
    )
    return audio


def _audio_feature_cube(
    audio_feature_long: pd.DataFrame | None,
    datetimes: pd.Index,
    ordered_nodes: list[int],
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    if audio_feature_long is None or len(feature_names) == 0:
        return np.zeros((len(datetimes), len(ordered_nodes), 0), dtype=float), []

    audio = audio_feature_long.copy()
    audio["datetime"] = pd.to_datetime(audio["datetime"])
    audio["node"] = audio["node"].astype(int)

    available = [name for name in feature_names if name in audio.columns]
    cube = np.zeros((len(datetimes), len(ordered_nodes), len(available)), dtype=float)

    for k, name in enumerate(available):
        pivot = audio.pivot_table(
            index="datetime",
            columns="node",
            values=name,
            aggfunc="first",
        )
        pivot = pivot.reindex(index=pd.DatetimeIndex(datetimes), columns=ordered_nodes)
        cube[:, :, k] = _clean_matrix(pivot.to_numpy(dtype=float))

    return cube, available


def build_processed_two_tower_data(
    gdf_cleaned: pd.DataFrame,
    gdf_nodes: pd.DataFrame,
    valid_indices: list[int],
    node_list: list[int],
    vehicle_list: list[str] | None = None,
    audio_feature_long: pd.DataFrame | None = None,
    *,
    history_steps: int = 5,
    max_subset_size: int = 3,
    utility_second_weight: float = 0.45,
    utility_third_weight: float = 0.20,
    rho: float | None = None,
    softmax_temperature: float = 4.0,
    context_audio_features: list[str] | None = None,
    action_audio_features: list[str] | None = None,
    include_audio_derived_features: bool = True,
    verbose: bool = True,
    progress_every: int = 1000,
    vehicle_id: str | None = None,
) -> dict[str, Any]:
    """
    Aggregate reference-style RSSI/geometry features plus extracted FLAC features.
    Compatible with both single and multi-vehicle dataloaders.

    Args:
        gdf_cleaned: GeoDataFrame with RSSI columns, per-node distance columns,
                     and optionally per-vehicle geometry columns (geometry_vehicle1, geometry_vehicle2, ...).
        gdf_nodes: GeoDataFrame with sensor node metadata.
        valid_indices: List of valid time indices.
        node_list: List of node IDs to process.
        vehicle_list: Optional list of vehicle IDs (e.g., ['vehicle1', 'vehicle2']) to process.
                     If provided, processes each vehicle separately and combines results.
                     If None or empty, processes a single vehicle using vehicle_id parameter.
        audio_feature_long: Optional DataFrame with per-node audio features.
        history_steps: Number of historical steps for rolling statistics.
        max_subset_size: Maximum subset size for multi-node combinations.
        utility_second_weight: Weight for second-closest distance in utility function.
        utility_third_weight: Weight for third-closest distance in utility function.
        rho: Distance scale parameter for utility (auto-computed from data if None).
        softmax_temperature: Temperature for softmax normalization of energy shares.
        context_audio_features: List of audio feature names for context vectors.
        action_audio_features: List of audio feature names for action vectors.
        include_audio_derived_features: Whether to compute derived audio score features.
        verbose: Whether to print progress.
        progress_every: Print progress every N timesteps.
        vehicle_id: Optional vehicle ID (e.g., 'vehicle1') for single-vehicle case.
                   If None and vehicle_list is empty, uses first available detected vehicle.

    Returns:
        Dictionary with keys:
            - feature_wide_df: Aggregated features across all nodes
            - node_feature_df: Per-node features at each timestep
            - ground_truth_node_df: Per-node distance and rank information
            - ground_truth_vehicle_df: Per-vehicle location and closest node (for multi-vehicle, includes 'vehicle' column)
            - sensor_geometry_df: Static sensor node positions
            - sequence_df: Context vectors and metadata for each timestep
            - examples_df: Training examples with actions and utility labels
            - meta: Metadata dictionary

    Note:
        Feature artifacts intentionally exclude vehicle coordinates and distances.
        Ground-truth vehicle geometry and sensor distances are saved separately.
    """
    if not valid_indices:
        raise ValueError("valid_indices is empty. Run the first dataloader cell first.")
    if "Node #" not in gdf_nodes.columns:
        raise ValueError('gdf_nodes must contain a "Node #" column.')

    context_audio_features = (
        DEFAULT_CONTEXT_AUDIO_FEATURES if context_audio_features is None else context_audio_features
    )
    action_audio_features = (
        DEFAULT_ACTION_AUDIO_FEATURES if action_audio_features is None else action_audio_features
    )
    if include_audio_derived_features:
        audio_feature_long = _add_audio_signature_scores(audio_feature_long)

    seq = gdf_cleaned.iloc[valid_indices].copy().sort_index()
    
    # Detect vehicles if not specified
    detected_vehicles = _detect_vehicles(seq)
    
    # Handle vehicle_list: process multiple vehicles separately and combine
    if vehicle_list and len(vehicle_list) > 0:
        if verbose:
            print(f"\nProcessing {len(vehicle_list)} vehicles separately...\n")
        
        all_feature_wide_dfs = []
        all_node_feature_dfs = []
        all_ground_truth_node_dfs = []
        all_ground_truth_vehicle_dfs = []
        all_sequence_dfs = []
        all_examples_dfs = []
        metadata_from_first = None
        
        for vid in vehicle_list:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Processing {vid.upper()}")
                print(f"{'='*60}\n")
            
            # Process this vehicle by calling the function recursively with single vehicle_id
            result = build_processed_two_tower_data(
                gdf_cleaned=gdf_cleaned,
                gdf_nodes=gdf_nodes,
                valid_indices=valid_indices,
                node_list=node_list,
                vehicle_list=None,  # Process one vehicle at a time
                audio_feature_long=audio_feature_long,
                history_steps=history_steps,
                max_subset_size=max_subset_size,
                utility_second_weight=utility_second_weight,
                utility_third_weight=utility_third_weight,
                rho=rho,
                softmax_temperature=softmax_temperature,
                context_audio_features=context_audio_features,
                action_audio_features=action_audio_features,
                include_audio_derived_features=include_audio_derived_features,
                verbose=verbose,
                progress_every=progress_every,
                vehicle_id=vid,
            )
            
            # Accumulate results
            all_feature_wide_dfs.append(result["feature_wide_df"].reset_index(names="datetime"))
            all_node_feature_dfs.append(result["node_feature_df"])
            all_ground_truth_node_dfs.append(result["ground_truth_node_df"])
            all_ground_truth_vehicle_dfs.append(result["ground_truth_vehicle_df"])
            all_sequence_dfs.append(result["sequence_df"])
            all_examples_dfs.append(result["examples_df"])
            
            if metadata_from_first is None:
                metadata_from_first = result["meta"]
        
        # Combine all results
        if verbose:
            print(f"\n{'='*60}")
            print("COMBINING ALL VEHICLES")
            print(f"{'='*60}\n")
        
        combined_processed = {
            "feature_wide_df": pd.concat(all_feature_wide_dfs, ignore_index=True).set_index("datetime").sort_index(),
            "node_feature_df": pd.concat(all_node_feature_dfs, ignore_index=True),
            "ground_truth_node_df": pd.concat(all_ground_truth_node_dfs, ignore_index=True),
            "ground_truth_vehicle_df": pd.concat(all_ground_truth_vehicle_dfs, ignore_index=True),
            "sensor_geometry_df": result["sensor_geometry_df"],  # Same for all vehicles
            "sequence_df": pd.concat(all_sequence_dfs, ignore_index=True),
            "examples_df": pd.concat(all_examples_dfs, ignore_index=True),
            "meta": metadata_from_first,
        }
        
        if verbose:
            print(f"Combined dataset sizes:")
            print(f"  feature_wide_df: {combined_processed['feature_wide_df'].shape}")
            print(f"  node_feature_df: {combined_processed['node_feature_df'].shape}")
            print(f"  ground_truth_node_df: {combined_processed['ground_truth_node_df'].shape}")
            print(f"  ground_truth_vehicle_df: {combined_processed['ground_truth_vehicle_df'].shape}")
            print(f"  sequence_df: {combined_processed['sequence_df'].shape}")
            print(f"  examples_df: {combined_processed['examples_df'].shape}")
        
        return combined_processed
    
    # Single vehicle case: use provided vehicle_id or first detected vehicle
    if not vehicle_id and detected_vehicles:
        vehicle_id = detected_vehicles[0]
    
    ordered_nodes = [
        int(node)
        for node in node_list
        if _node_col(int(node)) in seq.columns and _get_distance_columns(seq, int(node), vehicle_id) in seq.columns
    ]
    if not ordered_nodes:
        raise ValueError("No nodes have both RSSI and distance columns in gdf_cleaned.")

    T = len(seq)
    N = len(ordered_nodes)
    datetimes = pd.DatetimeIndex(seq.index)

    if verbose:
        print(f"valid timesteps: {T}")
        print(f"ordered nodes: {ordered_nodes}")
        if vehicle_id:
            print(f"using vehicle: {vehicle_id}")

    nodes_sub = gdf_nodes[gdf_nodes["Node #"].astype(int).isin(ordered_nodes)].copy()
    node_x = {int(row["Node #"]): float(row.geometry.x) for _, row in nodes_sub.iterrows()}
    node_y = {int(row["Node #"]): float(row.geometry.y) for _, row in nodes_sub.iterrows()}

    x_norm_map = _normalize_series(pd.Series(node_x)).to_dict()
    y_norm_map = _normalize_series(pd.Series(node_y)).to_dict()

    x_norm = np.array([x_norm_map[n] for n in ordered_nodes], dtype=float)
    y_norm = np.array([y_norm_map[n] for n in ordered_nodes], dtype=float)
    x_metric = np.array([node_x[n] for n in ordered_nodes], dtype=float)
    y_metric = np.array([node_y[n] for n in ordered_nodes], dtype=float)
    node_to_idx = {node: idx for idx, node in enumerate(ordered_nodes)}

    rssi_mat = _clean_matrix(
        np.stack([seq[_node_col(n)].to_numpy(dtype=float) for n in ordered_nodes], axis=1)
    )
    dist_mat = _clean_matrix(
        np.stack([seq[_get_distance_columns(seq, n, vehicle_id)].to_numpy(dtype=float) for n in ordered_nodes], axis=1)
    )

    audio_cube_context, context_audio_available = _audio_feature_cube(
        audio_feature_long,
        datetimes,
        ordered_nodes,
        context_audio_features,
    )
    audio_cube_action, action_audio_available = _audio_feature_cube(
        audio_feature_long,
        datetimes,
        ordered_nodes,
        action_audio_features,
    )
    audio_cube_scores, score_features_available = _audio_feature_cube(
        audio_feature_long,
        datetimes,
        ordered_nodes,
        DERIVED_AUDIO_SCORE_FEATURES if include_audio_derived_features else [],
    )
    global_audio_feature_request = [
        "construction_like_score",
        "vehicle_like_score",
        "spectral_entropy",
        "crest_factor_db",
        "band_2000_6000_ratio",
    ]
    audio_cube_global, global_audio_available = _audio_feature_cube(
        audio_feature_long,
        datetimes,
        ordered_nodes,
        global_audio_feature_request if include_audio_derived_features else [],
    )

    rssi_mean = np.zeros((T, N), dtype=float)
    rssi_std = np.zeros((T, N), dtype=float)
    rssi_median = np.zeros((T, N), dtype=float)
    rssi_mad = np.zeros((T, N), dtype=float)
    rssi_iqr = np.zeros((T, N), dtype=float)
    rssi_trimmed_mean = np.zeros((T, N), dtype=float)
    rssi_slope = np.zeros((T, N), dtype=float)
    rssi_delta1 = np.zeros((T, N), dtype=float)
    rssi_persistence_top3 = np.zeros((T, N), dtype=float)
    rssi_persistence_top5 = np.zeros((T, N), dtype=float)
    energy_share = np.zeros((T, N), dtype=float)
    rank_percentile = np.zeros((T, N), dtype=float)
    signal_rank = np.zeros((T, N), dtype=int)

    for t in range(T):
        if verbose and (t == 0 or (t + 1) % progress_every == 0 or t == T - 1):
            print(f"temporal features: {t + 1}/{T}")

        current_vals = rssi_mat[t]
        shares = _softmax(current_vals, temperature=softmax_temperature)
        energy_share[t] = shares

        order_desc = np.argsort(-current_vals)
        rank_pos = np.empty(N, dtype=int)
        rank_pos[order_desc] = np.arange(N)
        signal_rank[t] = rank_pos + 1
        rank_percentile[t] = 1.0 - (rank_pos / max(N - 1, 1))

        h0 = max(0, t - history_steps + 1)
        hist = rssi_mat[h0 : t + 1]
        rssi_mean[t] = hist.mean(axis=0)
        rssi_std[t] = hist.std(axis=0, ddof=0)
        rssi_median[t] = np.median(hist, axis=0)

        for j in range(N):
            hcol = hist[:, j]
            rssi_mad[t, j] = _mad(hcol)
            rssi_iqr[t, j] = _iqr(hcol)
            rssi_trimmed_mean[t, j] = _trimmed_mean(hcol)
            rssi_slope[t, j] = _linear_slope(hcol)
            rssi_delta1[t, j] = float(hcol[-1] - hcol[-2]) if len(hcol) >= 2 else 0.0

        hist_ranks = np.zeros_like(hist, dtype=int)
        for h in range(hist.shape[0]):
            hist_order = np.argsort(-hist[h])
            hist_pos = np.empty(N, dtype=int)
            hist_pos[hist_order] = np.arange(1, N + 1)
            hist_ranks[h] = hist_pos
        rssi_persistence_top3[t] = np.mean(hist_ranks <= min(3, N), axis=0)
        rssi_persistence_top5[t] = np.mean(hist_ranks <= min(5, N), axis=0)

    nearest_dists = np.min(dist_mat, axis=1)
    if rho is None:
        finite_nearest = nearest_dists[np.isfinite(nearest_dists)]
        rho = float(np.median(finite_nearest)) if len(finite_nearest) else 1.0
        rho = max(rho, 1.0)

    feature_wide_data: dict[str, np.ndarray] = {}
    node_feature_rows = []
    ground_truth_rows = []

    for j, node in enumerate(ordered_nodes):
        prefix = f"n{node}"
        feature_wide_data[f"{prefix}_rssi_db"] = rssi_mat[:, j]
        feature_wide_data[f"{prefix}_rssi_mean"] = rssi_mean[:, j]
        feature_wide_data[f"{prefix}_rssi_std"] = rssi_std[:, j]
        feature_wide_data[f"{prefix}_rssi_median"] = rssi_median[:, j]
        feature_wide_data[f"{prefix}_rssi_mad"] = rssi_mad[:, j]
        feature_wide_data[f"{prefix}_rssi_iqr"] = rssi_iqr[:, j]
        feature_wide_data[f"{prefix}_rssi_trimmed_mean"] = rssi_trimmed_mean[:, j]
        feature_wide_data[f"{prefix}_rssi_slope"] = rssi_slope[:, j]
        feature_wide_data[f"{prefix}_rssi_delta1"] = rssi_delta1[:, j]
        feature_wide_data[f"{prefix}_rssi_persistence_top3"] = rssi_persistence_top3[:, j]
        feature_wide_data[f"{prefix}_rssi_persistence_top5"] = rssi_persistence_top5[:, j]
        feature_wide_data[f"{prefix}_energy_share"] = energy_share[:, j]
        feature_wide_data[f"{prefix}_rank_percentile"] = rank_percentile[:, j]
        feature_wide_data[f"{prefix}_sensor_x_norm"] = np.full(T, x_norm[j], dtype=float)
        feature_wide_data[f"{prefix}_sensor_y_norm"] = np.full(T, y_norm[j], dtype=float)
        feature_wide_data[f"{prefix}_sensor_x_m"] = np.full(T, x_metric[j], dtype=float)
        feature_wide_data[f"{prefix}_sensor_y_m"] = np.full(T, y_metric[j], dtype=float)

        for k, name in enumerate(context_audio_available):
            feature_wide_data[f"{prefix}_audio_{name}"] = audio_cube_context[:, j, k]

        for k, name in enumerate(score_features_available):
            feature_wide_data[f"{prefix}_{name}"] = audio_cube_scores[:, j, k]

    feature_wide = pd.DataFrame(feature_wide_data, index=datetimes)

    for t, dt in enumerate(datetimes):
        if verbose and (t == 0 or (t + 1) % progress_every == 0 or t == T - 1):
            print(f"node rows: {t + 1}/{T}")

        dist_order = np.argsort(dist_mat[t])
        dist_rank = np.empty(N, dtype=int)
        dist_rank[dist_order] = np.arange(1, N + 1)

        for j, node in enumerate(ordered_nodes):
            base_features = {
                "rssi_db": rssi_mat[t, j],
                "rssi_mean": rssi_mean[t, j],
                "rssi_std": rssi_std[t, j],
                "rssi_median": rssi_median[t, j],
                "rssi_mad": rssi_mad[t, j],
                "rssi_iqr": rssi_iqr[t, j],
                "rssi_trimmed_mean": rssi_trimmed_mean[t, j],
                "rssi_slope": rssi_slope[t, j],
                "rssi_delta1": rssi_delta1[t, j],
                "rssi_persistence_top3": rssi_persistence_top3[t, j],
                "rssi_persistence_top5": rssi_persistence_top5[t, j],
                "energy_share": energy_share[t, j],
                "rank_percentile": rank_percentile[t, j],
                "sensor_x_norm": x_norm[j],
                "sensor_y_norm": y_norm[j],
                "sensor_x_m": x_metric[j],
                "sensor_y_m": y_metric[j],
            }

            row = {"datetime": dt, "node": node}
            row.update({name: float(value) for name, value in base_features.items()})

            for k, name in enumerate(context_audio_available):
                value = float(audio_cube_context[t, j, k])
                row[f"audio_{name}"] = value

            for k, name in enumerate(score_features_available):
                row[name] = float(audio_cube_scores[t, j, k])

            node_feature_rows.append(row)
            ground_truth_rows.append(
                {
                    "datetime": dt,
                    "node": node,
                    "distance_to_vehicle_m": float(dist_mat[t, j]),
                    "distance_rank": int(dist_rank[j]),
                    "is_closest": int(dist_rank[j] == 1),
                    "is_top2_closest": int(dist_rank[j] <= 2),
                    "is_top3_closest": int(dist_rank[j] <= 3),
                }
            )

    entropy = -np.sum(energy_share * np.log(energy_share + 1e-12), axis=1)
    order_by_share = np.argsort(-energy_share, axis=1)
    top1_share = energy_share[np.arange(T), order_by_share[:, 0]]
    top2_share = energy_share[np.arange(T), order_by_share[:, 1]] if N >= 2 else np.zeros(T)
    top3_share = energy_share[np.arange(T), order_by_share[:, min(2, N - 1)]]
    top5_share = energy_share[np.arange(T), order_by_share[:, min(4, N - 1)]]
    signal_top1_nodes = np.array([ordered_nodes[i] for i in order_by_share[:, 0]], dtype=int)
    closest_nodes = np.array([ordered_nodes[i] for i in np.argmin(dist_mat, axis=1)], dtype=int)

    acoustic_com_x = np.sum(energy_share * x_norm[None, :], axis=1)
    acoustic_com_y = np.sum(energy_share * y_norm[None, :], axis=1)

    feature_wide["rssi_entropy"] = entropy
    feature_wide["rssi_top1_top2_gap"] = top1_share - top2_share
    feature_wide["rssi_top1_top3_gap"] = top1_share - top3_share
    feature_wide["rssi_top1_top5_gap"] = top1_share - top5_share
    feature_wide["global_rssi_mean"] = np.mean(rssi_mat, axis=1)
    feature_wide["global_rssi_std"] = np.std(rssi_mat, axis=1, ddof=0)
    feature_wide["global_rssi_range"] = np.max(rssi_mat, axis=1) - np.min(rssi_mat, axis=1)
    feature_wide["signal_top1_node"] = signal_top1_nodes
    feature_wide["acoustic_com_x_norm"] = acoustic_com_x
    feature_wide["acoustic_com_y_norm"] = acoustic_com_y

    global_audio_stats: dict[str, np.ndarray] = {}
    for k, name in enumerate(global_audio_available):
        values = audio_cube_global[:, :, k]
        for stat_name, stat_values in {
            "mean": np.mean(values, axis=1),
            "max": np.max(values, axis=1),
            "std": np.std(values, axis=1, ddof=0),
            "range": np.max(values, axis=1) - np.min(values, axis=1),
        }.items():
            col = f"global_{stat_name}_{name}"
            global_audio_stats[col] = stat_values
            feature_wide[col] = stat_values

    # === Handle multi-vehicle ground truth ===
    vehicle_labels = _detect_vehicles(seq)
    ground_truth_vehicle_rows = []
    
    if vehicle_labels:
        # Multi-vehicle case
        for vehicle_label in vehicle_labels:
            geom_col = f"geometry_{vehicle_label}"
            if geom_col in seq.columns:
                for t, dt in enumerate(datetimes):
                    geom = seq.iloc[t][geom_col]
                    ground_truth_vehicle_rows.append(
                        {
                            "datetime": dt,
                            "vehicle": vehicle_label,
                            "vehicle_x_m": float(geom.x) if geom is not None else np.nan,
                            "vehicle_y_m": float(geom.y) if geom is not None else np.nan,
                            "closest_node": int(closest_nodes[t]),
                            "nearest_distance_m": float(nearest_dists[t]),
                        }
                    )
    else:
        # Single vehicle case (fallback if no vehicle-specific geometry)
        if "geometry" in seq.columns:
            for t, dt in enumerate(datetimes):
                geom = seq.iloc[t]["geometry"]
                ground_truth_vehicle_rows.append(
                    {
                        "datetime": dt,
                        "vehicle_x_m": float(geom.x),
                        "vehicle_y_m": float(geom.y),
                        "closest_node": int(closest_nodes[t]),
                        "nearest_distance_m": float(nearest_dists[t]),
                    }
                )
    
    ground_truth_vehicle_df = pd.DataFrame(ground_truth_vehicle_rows)

    node_feature_df = pd.DataFrame(node_feature_rows)
    ground_truth_node_df = pd.DataFrame(ground_truth_rows)

    full_subset_universe: list[tuple[int, ...]] = []
    for k in range(1, min(max_subset_size, N) + 1):
        full_subset_universe.extend(list(combinations(ordered_nodes, k)))

    subset_node_ids = []
    subset_node_idxs = []
    subset_strs = []
    subset_sizes = []
    subset_masks = []
    subset_geometry_static = []

    for subset in full_subset_universe:
        subset = tuple(sorted(subset))
        idxs = np.array([node_to_idx[n] for n in subset], dtype=int)
        subset_node_ids.append(subset)
        subset_node_idxs.append(idxs)
        subset_strs.append(_subset_to_str(subset))
        subset_sizes.append(len(subset))

        mask = np.zeros(N, dtype=float)
        mask[idxs] = 1.0
        subset_masks.append(mask)

        pairwise_d = []
        for a, b in combinations(idxs, 2):
            dx = x_metric[a] - x_metric[b]
            dy = y_metric[a] - y_metric[b]
            pairwise_d.append(float(np.sqrt(dx * dx + dy * dy)))
        while len(pairwise_d) < 3:
            pairwise_d.append(0.0)

        centroid_x = float(np.mean(x_norm[idxs]))
        centroid_y = float(np.mean(y_norm[idxs]))
        max_spread = float(max(pairwise_d)) if pairwise_d else 0.0
        tri_area = _triangle_area([(x_norm[i], y_norm[i]) for i in idxs])
        subset_geometry_static.append(pairwise_d[:3] + [centroid_x, centroid_y, max_spread, tri_area])

    selected_base_names = [
        "sensor_x_norm",
        "sensor_y_norm",
        "rssi_db",
        "rssi_mean",
        "rssi_median",
        "rssi_mad",
        "rssi_iqr",
        "rssi_trimmed_mean",
        "rssi_slope",
        "rssi_persistence_top3",
        "rssi_persistence_top5",
        "energy_share",
        "rank_percentile",
        "signal_rank",
    ]
    selected_score_names = list(score_features_available)
    selected_desc_width = len(selected_base_names) + len(action_audio_available)
    selected_desc_width += len(selected_score_names)
    selected_desc_len = max_subset_size * selected_desc_width
    geometry_names = [
        "pairwise_dist_1_m",
        "pairwise_dist_2_m",
        "pairwise_dist_3_m",
        "subset_centroid_x_norm",
        "subset_centroid_y_norm",
        "subset_max_spread_m",
        "subset_triangle_area_norm",
        "subset_centroid_minus_acoustic_com_x",
        "subset_centroid_minus_acoustic_com_y",
        "subset_centroid_to_acoustic_com_dist",
    ]
    geometry_len = len(geometry_names)
    acoustic_agg_names = [
        "subset_size",
        "sum_share",
        "max_share",
        "min_share",
        "mean_rssi",
        "var_rssi",
        "mean_slope",
        "has_signal_top1",
        "num_signal_top2_in_subset",
        "num_signal_top3_in_subset",
        "num_signal_top5_in_subset",
        "best_signal_rank_in_subset",
        "mean_signal_rank_in_subset",
        "worst_signal_rank_in_subset",
        "median_rssi",
        "mad_rssi",
        "iqr_rssi",
        "range_rssi",
    ]
    audio_agg_names = [
        f"{stat}_audio_{name}"
        for name in action_audio_available
        for stat in ("mean", "max", "min", "std", "range")
    ]
    score_agg_names = [
        f"{stat}_{name}"
        for name in score_features_available
        for stat in ("mean", "max", "std")
    ]

    action_layout = {}
    cursor = 0
    action_layout["membership_mask"] = [cursor, cursor + N]
    cursor += N
    action_layout["selected_desc"] = [cursor, cursor + selected_desc_len]
    cursor += selected_desc_len
    action_layout["subset_geometry"] = [cursor, cursor + geometry_len]
    cursor += geometry_len
    action_layout["acoustic_agg"] = [cursor, cursor + len(acoustic_agg_names)]
    cursor += len(acoustic_agg_names)
    action_layout["audio_agg"] = [cursor, cursor + len(audio_agg_names)]
    cursor += len(audio_agg_names)
    action_layout["score_agg"] = [cursor, cursor + len(score_agg_names)]
    cursor += len(score_agg_names)

    action_feature_names = []
    action_feature_names.extend([f"mask_n{node}" for node in ordered_nodes])
    selected_names = selected_base_names + [f"audio_{name}" for name in action_audio_available] + selected_score_names
    for slot in range(1, max_subset_size + 1):
        action_feature_names.extend([f"slot{slot}_{name}" for name in selected_names])
    action_feature_names.extend(geometry_names)
    action_feature_names.extend(acoustic_agg_names)
    action_feature_names.extend(audio_agg_names)
    action_feature_names.extend(score_agg_names)

    sequence_rows = []
    examples_rows = []
    subset_sizes_arr = np.array(subset_sizes, dtype=int)

    for t, dt in enumerate(datetimes):
        if verbose and (t == 0 or (t + 1) % progress_every == 0 or t == T - 1):
            print(f"action rows: {t + 1}/{T}")

        shares_t = energy_share[t]
        top1_idx = int(order_by_share[t, 0])

        context_vec = []
        context_names = []
        for j, node in enumerate(ordered_nodes):
            values = [
                rssi_mat[t, j],
                rssi_mean[t, j],
                rssi_std[t, j],
                rssi_median[t, j],
                rssi_mad[t, j],
                rssi_iqr[t, j],
                rssi_trimmed_mean[t, j],
                rssi_slope[t, j],
                rssi_delta1[t, j],
                rssi_persistence_top3[t, j],
                rssi_persistence_top5[t, j],
                energy_share[t, j],
                rank_percentile[t, j],
                x_norm[j],
                y_norm[j],
            ]
            names = [
                "rssi_db",
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
                "energy_share",
                "rank_percentile",
                "sensor_x_norm",
                "sensor_y_norm",
            ]
            for k, name in enumerate(context_audio_available):
                values.append(audio_cube_context[t, j, k])
                names.append(f"audio_{name}")

            for k, name in enumerate(score_features_available):
                values.append(audio_cube_scores[t, j, k])
                names.append(name)

            context_vec.extend([float(v) for v in values])
            context_names.extend([f"n{node}_{name}" for name in names])

        global_values = [
            entropy[t],
            top1_share[t] - top2_share[t],
            top1_share[t] - top3_share[t],
            top1_share[t] - top5_share[t],
            float(np.mean(rssi_mat[t])),
            float(np.std(rssi_mat[t], ddof=0)),
            float(np.max(rssi_mat[t]) - np.min(rssi_mat[t])),
            acoustic_com_x[t],
            acoustic_com_y[t],
        ]
        global_names = [
            "rssi_entropy",
            "rssi_top1_top2_gap",
            "rssi_top1_top3_gap",
            "rssi_top1_top5_gap",
            "global_rssi_mean",
            "global_rssi_std",
            "global_rssi_range",
            "acoustic_com_x_norm",
            "acoustic_com_y_norm",
        ]
        for name, values in global_audio_stats.items():
            global_values.append(values[t])
            global_names.append(name)
        context_vec.extend([float(v) for v in global_values])
        context_names.extend(global_names)
        context_vec = np.asarray(context_vec, dtype=np.float32)

        sequence_rows.append(
            {
                "time_id": t,
                "datetime": dt,
                "context_vec": context_vec,
                "num_actions": len(full_subset_universe),
                "signal_top1_node": int(signal_top1_nodes[t]),
            }
        )

        for m, idxs in enumerate(subset_node_idxs):
            subset = subset_node_ids[m]
            sort_local = idxs[np.argsort(-shares_t[idxs])]

            selected_desc = []
            for j in sort_local:
                selected_desc.extend(
                    [
                        float(x_norm[j]),
                        float(y_norm[j]),
                        float(rssi_mat[t, j]),
                        float(rssi_mean[t, j]),
                        float(rssi_median[t, j]),
                        float(rssi_mad[t, j]),
                        float(rssi_iqr[t, j]),
                        float(rssi_trimmed_mean[t, j]),
                        float(rssi_slope[t, j]),
                        float(rssi_persistence_top3[t, j]),
                        float(rssi_persistence_top5[t, j]),
                        float(energy_share[t, j]),
                        float(rank_percentile[t, j]),
                        float(signal_rank[t, j]),
                    ]
                )
                selected_desc.extend([float(v) for v in audio_cube_action[t, j, :]])
                selected_desc.extend([float(v) for v in audio_cube_scores[t, j, :]])
            while len(selected_desc) < selected_desc_len:
                selected_desc.append(0.0)

            subset_shares = shares_t[idxs]
            subset_rssi_vals = rssi_mat[t, idxs]
            subset_slopes = rssi_slope[t, idxs]
            subset_signal_ranks = signal_rank[t, idxs]
            subset_audio = audio_cube_action[t, idxs, :]
            subset_scores = audio_cube_scores[t, idxs, :]
            centroid_x = subset_geometry_static[m][3]
            centroid_y = subset_geometry_static[m][4]
            dx_acoustic = float(centroid_x - acoustic_com_x[t])
            dy_acoustic = float(centroid_y - acoustic_com_y[t])
            subset_geometry = subset_geometry_static[m] + [
                dx_acoustic,
                dy_acoustic,
                float(np.sqrt(dx_acoustic * dx_acoustic + dy_acoustic * dy_acoustic)),
            ]

            acoustic_agg = [
                float(len(idxs)),
                float(np.sum(subset_shares)),
                float(np.max(subset_shares)),
                float(np.min(subset_shares)),
                float(np.mean(subset_rssi_vals)),
                float(np.var(subset_rssi_vals)) if len(subset_rssi_vals) > 1 else 0.0,
                float(np.mean(subset_slopes)),
                float(1.0 if top1_idx in idxs else 0.0),
                float(np.sum(subset_signal_ranks <= min(2, N))),
                float(np.sum(subset_signal_ranks <= min(3, N))),
                float(np.sum(subset_signal_ranks <= min(5, N))),
                float(np.min(subset_signal_ranks)),
                float(np.mean(subset_signal_ranks)),
                float(np.max(subset_signal_ranks)),
                float(np.median(subset_rssi_vals)),
                float(_mad(subset_rssi_vals)),
                float(_iqr(subset_rssi_vals)),
                float(np.max(subset_rssi_vals) - np.min(subset_rssi_vals)),
            ]

            audio_agg = []
            for k in range(len(action_audio_available)):
                values = subset_audio[:, k]
                audio_agg.extend(
                    [
                        float(np.mean(values)),
                        float(np.max(values)),
                        float(np.min(values)),
                        float(np.std(values, ddof=0)),
                        float(np.max(values) - np.min(values)),
                    ]
                )

            score_agg = []
            for k in range(len(score_features_available)):
                values = subset_scores[:, k]
                score_agg.extend(
                    [
                        float(np.mean(values)),
                        float(np.max(values)),
                        float(np.std(values, ddof=0)),
                    ]
                )

            action_raw_vec = np.asarray(
                subset_masks[m].tolist()
                + selected_desc
                + subset_geometry
                + acoustic_agg
                + audio_agg
                + score_agg,
                dtype=np.float32,
            )

            dists = np.sort(dist_mat[t, idxs])
            d1 = float(dists[0])
            d2 = float(dists[1]) if len(dists) >= 2 else np.inf
            d3 = float(dists[2]) if len(dists) >= 3 else np.inf

            term1 = 1.0 / (1.0 + d1 / rho)
            term2 = utility_second_weight / (1.0 + d2 / rho) if np.isfinite(d2) else 0.0
            term3 = utility_third_weight / (1.0 + d3 / rho) if np.isfinite(d3) else 0.0
            utility = float(term1 + term2 + term3)

            subset_dist_ranks = [int(np.where(np.argsort(dist_mat[t]) == idx)[0][0] + 1) for idx in idxs]

            examples_rows.append(
                {
                    "time_id": t,
                    "datetime": dt,
                    "subset": subset,
                    "subset_str": subset_strs[m],
                    "subset_size": int(subset_sizes_arr[m]),
                    "context_vec": context_vec,
                    "action_raw_vec": action_raw_vec,
                    "utility": utility,
                    "d1": d1,
                    "d2": float(d2) if np.isfinite(d2) else np.nan,
                    "d3": float(d3) if np.isfinite(d3) else np.nan,
                    "contains_closest_node": int(np.argmin(dist_mat[t]) in idxs),
                    "best_rank_in_subset": int(min(subset_dist_ranks)),
                    "signal_top1_node": int(signal_top1_nodes[t]),
                    "closest_node": int(closest_nodes[t]),
                }
            )

    sequence_df = pd.DataFrame(sequence_rows).sort_values("datetime").reset_index(drop=True)
    examples_df = (
        pd.DataFrame(examples_rows)
        .sort_values(["datetime", "subset_size", "subset_str"])
        .reset_index(drop=True)
    )

    sensor_geometry_df = pd.DataFrame(
        {
            "node": ordered_nodes,
            "sensor_x_m": x_metric,
            "sensor_y_m": y_metric,
            "sensor_x_norm": x_norm,
            "sensor_y_norm": y_norm,
        }
    )

    meta = {
        "ordered_nodes": ordered_nodes,
        "vehicle_id": vehicle_id,
        "detected_vehicles": detected_vehicles,
        "history_steps": history_steps,
        "max_subset_size": max_subset_size,
        "utility_second_weight": utility_second_weight,
        "utility_third_weight": utility_third_weight,
        "rho": rho,
        "softmax_temperature": softmax_temperature,
        "context_dim": int(len(sequence_df.iloc[0]["context_vec"])),
        "action_raw_dim": int(len(examples_df.iloc[0]["action_raw_vec"])),
        "num_times": int(len(sequence_df)),
        "num_examples": int(len(examples_df)),
        "num_actions_per_time": int(len(full_subset_universe)),
        "vehicle_labels": vehicle_labels if vehicle_labels else ["vehicle"],
        "context_feature_names": context_names,
        "action_feature_names": action_feature_names,
        "per_node_rssi_feature_names": [
            "rssi_median",
            "rssi_mad",
            "rssi_iqr",
            "rssi_trimmed_mean",
            "rssi_persistence_top3",
            "rssi_persistence_top5",
        ],
        "selected_desc_base_names": selected_base_names,
        "selected_desc_score_names": selected_score_names,
        "context_audio_features": context_audio_available,
        "action_audio_features": action_audio_available,
        "derived_audio_score_features": score_features_available,
        "global_audio_features": global_audio_available,
        "global_feature_names": global_names,
        "subset_geometry_names": geometry_names,
        "acoustic_agg_names": acoustic_agg_names,
        "audio_agg_names": audio_agg_names,
        "score_agg_names": score_agg_names,
        "action_layout": action_layout,
        "full_subset_universe": full_subset_universe,
        "leakage_note": (
            "feature_wide_df, node_feature_df, context_vec, and action_raw_vec exclude "
            "vehicle coordinates and distance-to-vehicle targets. Target columns are saved "
            "only in ground_truth_* and examples_df label columns. For multi-vehicle case, "
            "ground_truth_vehicle_df has a 'vehicle' column identifying each vehicle."
        ),
    }

    if verbose:
        print("processed two-tower data ready")
        print(f"sequence_df: {sequence_df.shape}")
        print(f"examples_df: {examples_df.shape}")
        print(f"context_dim: {meta['context_dim']}")
        print(f"action_raw_dim: {meta['action_raw_dim']}")

    return {
        "feature_wide_df": feature_wide,
        "node_feature_df": node_feature_df,
        "ground_truth_node_df": ground_truth_node_df,
        "ground_truth_vehicle_df": ground_truth_vehicle_df,
        "sensor_geometry_df": sensor_geometry_df,
        "sequence_df": sequence_df,
        "examples_df": examples_df,
        "meta": meta,
    }


def save_processed_two_tower_data(
    processed: dict[str, Any],
    processed_dir: str | Path,
    *,
    prefix: str = "two_tower",
) -> dict[str, Path]:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "feature_wide_csv": processed_dir / f"{prefix}_feature_wide.csv",
        "node_features_csv": processed_dir / f"{prefix}_node_features.csv",
        "ground_truth_nodes_csv": processed_dir / f"{prefix}_ground_truth_nodes.csv",
        "ground_truth_vehicle_csv": processed_dir / f"{prefix}_ground_truth_vehicle.csv",
        "sensor_geometry_csv": processed_dir / f"{prefix}_sensor_geometry.csv",
        "sequence_pkl": processed_dir / f"{prefix}_sequence.pkl",
        "examples_pkl": processed_dir / f"{prefix}_examples.pkl",
        "examples_index_csv": processed_dir / f"{prefix}_examples_index.csv",
        "arrays_npz": processed_dir / f"{prefix}_arrays.npz",
        "meta_json": processed_dir / f"{prefix}_meta.json",
    }

    processed["feature_wide_df"].reset_index(names="datetime").to_csv(paths["feature_wide_csv"], index=False)
    processed["node_feature_df"].to_csv(paths["node_features_csv"], index=False)
    processed["ground_truth_node_df"].to_csv(paths["ground_truth_nodes_csv"], index=False)
    processed["ground_truth_vehicle_df"].to_csv(paths["ground_truth_vehicle_csv"], index=False)
    processed["sensor_geometry_df"].to_csv(paths["sensor_geometry_csv"], index=False)

    processed["sequence_df"].to_pickle(paths["sequence_pkl"])
    processed["examples_df"].to_pickle(paths["examples_pkl"])

    examples_index = processed["examples_df"].drop(columns=["context_vec", "action_raw_vec"])
    examples_index.to_csv(paths["examples_index_csv"], index=False)

    C_by_time = np.stack(processed["sequence_df"]["context_vec"].to_numpy()).astype(np.float32)
    A_examples = np.stack(processed["examples_df"]["action_raw_vec"].to_numpy()).astype(np.float32)
    y_examples = processed["examples_df"]["utility"].to_numpy(dtype=np.float32)
    example_time_id = processed["examples_df"]["time_id"].to_numpy(dtype=np.int32)
    example_subset_size = processed["examples_df"]["subset_size"].to_numpy(dtype=np.int16)
    example_d1 = processed["examples_df"]["d1"].to_numpy(dtype=np.float32)
    example_contains_closest = processed["examples_df"]["contains_closest_node"].to_numpy(dtype=np.int8)

    np.savez_compressed(
        paths["arrays_npz"],
        C_by_time=C_by_time,
        A_examples=A_examples,
        y_examples=y_examples,
        example_time_id=example_time_id,
        example_subset_size=example_subset_size,
        example_d1=example_d1,
        example_contains_closest=example_contains_closest,
        sequence_times=processed["sequence_df"]["datetime"].to_numpy(dtype="datetime64[ns]"),
    )

    with open(paths["meta_json"], "w", encoding="utf-8") as f:
        json.dump(_as_jsonable(processed["meta"]), f, indent=2)

    return paths
