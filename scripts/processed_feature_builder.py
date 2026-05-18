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

DEFAULT_ACTION_AUDIO_FEATURES = [
    "rms_db",
    "spectral_centroid_hz",
    "spectral_flatness",
    "band_20_120_ratio",
    "band_120_500_ratio",
    "low_to_voice_db",
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
    verbose: bool = True,
    progress_every: int = 1000,
) -> dict[str, Any]:
    """
    Aggregate reference-style RSSI/geometry features plus extracted FLAC features.

    Feature artifacts intentionally exclude vehicle coordinates and distances.
    Ground-truth vehicle geometry and sensor distances are saved separately.
    """
    if not valid_indices:
        raise ValueError("valid_indices is empty. Run the first dataloader cell first.")
    if "Node #" not in gdf_nodes.columns:
        raise ValueError('gdf_nodes must contain a "Node #" column.')

    context_audio_features = context_audio_features or DEFAULT_CONTEXT_AUDIO_FEATURES
    action_audio_features = action_audio_features or DEFAULT_ACTION_AUDIO_FEATURES

    seq = gdf_cleaned.iloc[valid_indices].copy().sort_index()
    ordered_nodes = [
        int(node)
        for node in node_list
        if _node_col(int(node)) in seq.columns and _dist_col(int(node)) in seq.columns
    ]
    if not ordered_nodes:
        raise ValueError("No nodes have both RSSI and distance columns in gdf_cleaned.")

    T = len(seq)
    N = len(ordered_nodes)
    datetimes = pd.DatetimeIndex(seq.index)

    if verbose:
        print(f"valid timesteps: {T}")
        print(f"ordered nodes: {ordered_nodes}")

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
        np.stack([seq[_dist_col(n)].to_numpy(dtype=float) for n in ordered_nodes], axis=1)
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

    rssi_mean = np.zeros((T, N), dtype=float)
    rssi_std = np.zeros((T, N), dtype=float)
    rssi_slope = np.zeros((T, N), dtype=float)
    rssi_delta1 = np.zeros((T, N), dtype=float)
    energy_share = np.zeros((T, N), dtype=float)
    rank_percentile = np.zeros((T, N), dtype=float)

    for t in range(T):
        if verbose and (t == 0 or (t + 1) % progress_every == 0 or t == T - 1):
            print(f"temporal features: {t + 1}/{T}")

        current_vals = rssi_mat[t]
        shares = _softmax(current_vals, temperature=softmax_temperature)
        energy_share[t] = shares

        order_desc = np.argsort(-current_vals)
        rank_pos = np.empty(N, dtype=int)
        rank_pos[order_desc] = np.arange(N)
        rank_percentile[t] = 1.0 - (rank_pos / max(N - 1, 1))

        h0 = max(0, t - history_steps + 1)
        hist = rssi_mat[h0 : t + 1]
        rssi_mean[t] = hist.mean(axis=0)
        rssi_std[t] = hist.std(axis=0, ddof=0)

        for j in range(N):
            hcol = hist[:, j]
            rssi_slope[t, j] = _linear_slope(hcol)
            rssi_delta1[t, j] = float(hcol[-1] - hcol[-2]) if len(hcol) >= 2 else 0.0

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
        feature_wide_data[f"{prefix}_rssi_slope"] = rssi_slope[:, j]
        feature_wide_data[f"{prefix}_rssi_delta1"] = rssi_delta1[:, j]
        feature_wide_data[f"{prefix}_energy_share"] = energy_share[:, j]
        feature_wide_data[f"{prefix}_rank_percentile"] = rank_percentile[:, j]
        feature_wide_data[f"{prefix}_sensor_x_norm"] = np.full(T, x_norm[j], dtype=float)
        feature_wide_data[f"{prefix}_sensor_y_norm"] = np.full(T, y_norm[j], dtype=float)
        feature_wide_data[f"{prefix}_sensor_x_m"] = np.full(T, x_metric[j], dtype=float)
        feature_wide_data[f"{prefix}_sensor_y_m"] = np.full(T, y_metric[j], dtype=float)

        for k, name in enumerate(context_audio_available):
            feature_wide_data[f"{prefix}_audio_{name}"] = audio_cube_context[:, j, k]

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
                "rssi_slope": rssi_slope[t, j],
                "rssi_delta1": rssi_delta1[t, j],
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
    signal_top1_nodes = np.array([ordered_nodes[i] for i in order_by_share[:, 0]], dtype=int)
    closest_nodes = np.array([ordered_nodes[i] for i in np.argmin(dist_mat, axis=1)], dtype=int)

    acoustic_com_x = np.sum(energy_share * x_norm[None, :], axis=1)
    acoustic_com_y = np.sum(energy_share * y_norm[None, :], axis=1)

    feature_wide["rssi_entropy"] = entropy
    feature_wide["rssi_top1_top2_gap"] = top1_share - top2_share
    feature_wide["signal_top1_node"] = signal_top1_nodes
    feature_wide["acoustic_com_x_norm"] = acoustic_com_x
    feature_wide["acoustic_com_y_norm"] = acoustic_com_y

    ground_truth_vehicle_df = pd.DataFrame(
        {
            "datetime": datetimes,
            "vehicle_x_m": [float(geom.x) for geom in seq.geometry],
            "vehicle_y_m": [float(geom.y) for geom in seq.geometry],
            "closest_node": closest_nodes,
            "nearest_distance_m": nearest_dists,
        }
    )

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
        "rssi_slope",
        "energy_share",
    ]
    selected_desc_width = len(selected_base_names) + len(action_audio_available)
    selected_desc_len = max_subset_size * selected_desc_width
    geometry_len = 7
    acoustic_agg_names = [
        "subset_size",
        "sum_share",
        "max_share",
        "min_share",
        "mean_rssi",
        "var_rssi",
        "mean_slope",
        "has_signal_top1",
    ]
    audio_agg_names = [
        f"{stat}_audio_{name}"
        for name in action_audio_available
        for stat in ("mean", "max")
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
                rssi_slope[t, j],
                rssi_delta1[t, j],
                energy_share[t, j],
                rank_percentile[t, j],
                x_norm[j],
                y_norm[j],
            ]
            names = [
                "rssi_db",
                "rssi_mean",
                "rssi_std",
                "rssi_slope",
                "rssi_delta1",
                "energy_share",
                "rank_percentile",
                "sensor_x_norm",
                "sensor_y_norm",
            ]
            for k, name in enumerate(context_audio_available):
                values.append(audio_cube_context[t, j, k])
                names.append(f"audio_{name}")

            context_vec.extend([float(v) for v in values])
            context_names.extend([f"n{node}_{name}" for name in names])

        global_values = [
            entropy[t],
            top1_share[t] - top2_share[t],
            acoustic_com_x[t],
            acoustic_com_y[t],
        ]
        global_names = [
            "rssi_entropy",
            "rssi_top1_top2_gap",
            "acoustic_com_x_norm",
            "acoustic_com_y_norm",
        ]
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
                        float(rssi_slope[t, j]),
                        float(energy_share[t, j]),
                    ]
                )
                selected_desc.extend([float(v) for v in audio_cube_action[t, j, :]])
            while len(selected_desc) < selected_desc_len:
                selected_desc.append(0.0)

            subset_shares = shares_t[idxs]
            subset_rssi_vals = rssi_mat[t, idxs]
            subset_slopes = rssi_slope[t, idxs]
            subset_audio = audio_cube_action[t, idxs, :]

            acoustic_agg = [
                float(len(idxs)),
                float(np.sum(subset_shares)),
                float(np.max(subset_shares)),
                float(np.min(subset_shares)),
                float(np.mean(subset_rssi_vals)),
                float(np.var(subset_rssi_vals)) if len(subset_rssi_vals) > 1 else 0.0,
                float(np.mean(subset_slopes)),
                float(1.0 if top1_idx in idxs else 0.0),
            ]

            audio_agg = []
            for k in range(len(action_audio_available)):
                values = subset_audio[:, k]
                audio_agg.extend([float(np.mean(values)), float(np.max(values))])

            action_raw_vec = np.asarray(
                subset_masks[m].tolist()
                + selected_desc
                + subset_geometry_static[m]
                + acoustic_agg
                + audio_agg,
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
        "context_feature_names": context_names,
        "selected_desc_base_names": selected_base_names,
        "context_audio_features": context_audio_available,
        "action_audio_features": action_audio_available,
        "acoustic_agg_names": acoustic_agg_names,
        "audio_agg_names": audio_agg_names,
        "action_layout": action_layout,
        "full_subset_universe": full_subset_universe,
        "leakage_note": (
            "feature_wide_df, node_feature_df, context_vec, and action_raw_vec exclude "
            "vehicle coordinates and distance-to-vehicle targets. Target columns are saved "
            "only in ground_truth_* and examples_df label columns."
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
