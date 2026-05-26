from __future__ import annotations

from pathlib import Path
import json
import gc
import math
import time
from typing import Any

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.graces_quarters_rssi_two_tower import (
    DEFAULT_MODEL_TAG,
    DEFAULT_PREFIX,
    _static_action_catalog_from_geometry,
)
from scripts.rssi_only_two_tower_baseline import build_per_time_decisions, fit_standardizer, predict_scores
from scripts.two_tower_training import TrainConfig, TwoTowerMLP, chronological_split


EPS = 1e-9
OUTPUT_SET_SIZE = 3
SENSORS_PER_CELL = 3
TOP_K_CELLS = 1
GRID_RADIUS_CELLS = 10
GRID_SPACING_M = 4.0
N_SPLINE_BINS = 8
FIT_MIN_POINTS_PER_NODE = 40
MIN_VARIANCE = 1e-6
KDE_BIN_WIDTH_M = 2.0
KDE_BIN_HALF_WIDTH_M = KDE_BIN_WIDTH_M / 2.0
KDE_MIN_SAMPLES = 5
KDE_SUPPORT_DELTA_M = 1.0
MIN_KDE_BANDWIDTH = 1e-3
KDE_LOOKUP_GRID_SIZE = 512


def _logdist(distance_m: np.ndarray | float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(distance_m, dtype=float), EPS))


def _gaussian_logpdf(z: float | np.ndarray, mu: np.ndarray, sigma2: np.ndarray | float) -> np.ndarray:
    sigma2 = np.maximum(np.asarray(sigma2, dtype=float), MIN_VARIANCE)
    return -0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * (np.asarray(z, dtype=float) - mu) ** 2 / sigma2


def _fit_pathloss(train_obs: pd.DataFrame, ordered_nodes: list[int]) -> dict[int, dict[str, Any]]:
    params: dict[int, dict[str, Any]] = {}
    for node in ordered_nodes:
        df = train_obs[train_obs["node"].eq(node)].dropna(subset=["rssi_db", "distance_to_vehicle_m"])
        df = df[df["distance_to_vehicle_m"].gt(0)]
        if len(df) < FIT_MIN_POINTS_PER_NODE:
            continue
        x = _logdist(df["distance_to_vehicle_m"].to_numpy(dtype=float))
        y = df["rssi_db"].to_numpy(dtype=float)
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        residual = y - pred
        params[int(node)] = {
            "P0": float(beta[0]),
            "eta": float(-beta[1]),
            "slope": float(beta[1]),
            "sigma2": max(float(np.var(residual)), MIN_VARIANCE),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "n_fit": int(len(df)),
            "distances": df["distance_to_vehicle_m"].to_numpy(dtype=float),
            "rssi": y,
        }
    return params


def _assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def _solve_cvxpy(problem: cp.Problem) -> str:
    attempts = [
        ("OSQP", dict(eps_abs=1e-6, eps_rel=1e-6, max_iter=100_000, polish=True)),
        ("CLARABEL", dict()),
        ("SCS", dict(eps=1e-5, max_iters=50_000)),
    ]
    last_error = None
    for solver, kwargs in attempts:
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, verbose=False, **kwargs)
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return solver
        except Exception as exc:  # pragma: no cover - solver fallback path
            last_error = exc
    raise RuntimeError(f"CVXPY spline solve failed; status={problem.status}, last_error={last_error}")


def _fit_spline(train_obs: pd.DataFrame, ordered_nodes: list[int]) -> dict[int, dict[str, Any]]:
    models: dict[int, dict[str, Any]] = {}
    for node in ordered_nodes:
        df = train_obs[train_obs["node"].eq(node)].dropna(subset=["rssi_db", "distance_to_vehicle_m"])
        df = df[df["distance_to_vehicle_m"].gt(0)]
        if len(df) < FIT_MIN_POINTS_PER_NODE:
            continue
        x = _logdist(df["distance_to_vehicle_m"].to_numpy(dtype=float))
        y = df["rssi_db"].to_numpy(dtype=float)
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            continue

        edges = np.linspace(x_min, x_max, N_SPLINE_BINS + 1)
        bin_id = _assign_bins(x, edges)
        a = cp.Variable(N_SPLINE_BINS)
        b = cp.Variable(N_SPLINE_BINS)
        obj_terms = []
        for w in range(N_SPLINE_BINS):
            idx = np.where(bin_id == w)[0]
            if len(idx):
                obj_terms.append(cp.sum_squares(y[idx] - (a[w] + b[w] * x[idx])))
        if not obj_terms:
            continue

        constraints = []
        for w in range(N_SPLINE_BINS - 1):
            boundary = float(edges[w + 1])
            constraints.append(a[w] + b[w] * boundary == a[w + 1] + b[w + 1] * boundary)
        objective = cp.sum(obj_terms) + 1e-6 * (cp.sum_squares(a) + cp.sum_squares(b))
        objective += 1e-6 * cp.sum_squares(b[1:] - b[:-1])
        problem = cp.Problem(cp.Minimize(objective), constraints)
        solver = _solve_cvxpy(problem)

        a_hat = np.asarray(a.value, dtype=float).reshape(-1)
        b_hat = np.asarray(b.value, dtype=float).reshape(-1)
        pred = a_hat[bin_id] + b_hat[bin_id] * x
        residual = y - pred
        global_var = max(float(np.var(residual)), MIN_VARIANCE)
        sigma2 = np.full(N_SPLINE_BINS, global_var, dtype=float)
        n_by_bin = np.zeros(N_SPLINE_BINS, dtype=int)
        for w in range(N_SPLINE_BINS):
            idx = np.where(bin_id == w)[0]
            n_by_bin[w] = len(idx)
            if len(idx) >= 3:
                sigma2[w] = max(float(np.var(residual[idx])), MIN_VARIANCE)
        models[int(node)] = {
            "a": a_hat,
            "b": b_hat,
            "edges": edges,
            "sigma2_by_bin": sigma2,
            "global_sigma2": global_var,
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "n_fit": int(len(df)),
            "n_by_bin": n_by_bin,
            "solver": solver,
            "distances": df["distance_to_vehicle_m"].to_numpy(dtype=float),
            "rssi": y,
        }
    return models


def _spline_predict(model: dict[str, Any], distance_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _logdist(distance_m)
    bin_id = _assign_bins(x, model["edges"])
    mu = model["a"][bin_id] + model["b"][bin_id] * x
    sigma2 = model["sigma2_by_bin"][bin_id]
    return mu, sigma2


def _fit_kde(train_obs: pd.DataFrame, ordered_nodes: list[int]) -> dict[int, list[dict[str, Any]]]:
    kde_models: dict[int, list[dict[str, Any]]] = {}
    for node in ordered_nodes:
        df = train_obs[train_obs["node"].eq(node)].dropna(subset=["rssi_db", "distance_to_vehicle_m"])
        if len(df) < FIT_MIN_POINTS_PER_NODE:
            continue
        d = df["distance_to_vehicle_m"].to_numpy(dtype=float)
        z = df["rssi_db"].to_numpy(dtype=float)
        d_min, d_max = float(np.min(d)), float(np.max(d))
        first = KDE_BIN_HALF_WIDTH_M + KDE_BIN_WIDTH_M * np.floor(d_min / KDE_BIN_WIDTH_M)
        last = KDE_BIN_HALF_WIDTH_M + KDE_BIN_WIDTH_M * np.ceil(d_max / KDE_BIN_WIDTH_M)
        centers = np.arange(first, last + KDE_BIN_WIDTH_M, KDE_BIN_WIDTH_M)
        bins = []
        for center in centers:
            keep = (d >= center - KDE_BIN_HALF_WIDTH_M) & (d < center + KDE_BIN_HALF_WIDTH_M)
            samples = z[keep]
            if len(samples) < KDE_MIN_SAMPLES:
                continue
            sigma = float(np.std(samples, ddof=1)) if len(samples) > 1 else MIN_KDE_BANDWIDTH
            bandwidth = max(sigma * (len(samples) ** (-1.0 / 5.0)), MIN_KDE_BANDWIDTH)
            bins.append({"center": float(center), "samples": samples.astype(float), "bandwidth": float(bandwidth)})
        if bins:
            kde_models[int(node)] = bins
    return kde_models


def _kde_logpdf(z: float, samples: np.ndarray, bandwidth: float) -> float:
    h = max(float(bandwidth), MIN_KDE_BANDWIDTH)
    u = (float(z) - samples) / h
    vals = -0.5 * u * u - np.log(h) - 0.5 * np.log(2.0 * np.pi)
    m = float(np.max(vals))
    return float(m + np.log(np.mean(np.exp(vals - m))))


def _kde_logpdf_lookup_grid(z_grid: np.ndarray, samples: np.ndarray, bandwidth: float) -> np.ndarray:
    h = max(float(bandwidth), MIN_KDE_BANDWIDTH)
    samples = np.asarray(samples, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    u = (z_grid[:, None] - samples[None, :]) / h
    vals = -0.5 * u * u - np.log(h) - 0.5 * np.log(2.0 * np.pi)
    m = np.max(vals, axis=1)
    return m + np.log(np.mean(np.exp(vals - m[:, None]), axis=1))


def _make_kde_lookup(samples: np.ndarray, bandwidth: float, n_grid: int = KDE_LOOKUP_GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    h = max(float(bandwidth), MIN_KDE_BANDWIDTH)
    lo = float(np.min(samples) - 6.0 * h)
    hi = float(np.max(samples) + 6.0 * h)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.mean(samples)) if len(samples) else 0.0
        lo, hi = center - 1.0, center + 1.0
    z_grid = np.linspace(lo, hi, int(n_grid), dtype=float)
    return z_grid, _kde_logpdf_lookup_grid(z_grid, samples, h)


def _make_kde_lookup_groups(
    kde_alloc: dict[int, list[dict[str, Any] | None]],
) -> dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for node, alloc in kde_alloc.items():
        by_model: dict[int, dict[str, Any]] = {}
        for gi, model in enumerate(alloc):
            if model is None:
                continue
            key = id(model)
            if key not in by_model:
                z_grid, logpdf_grid = _make_kde_lookup(model["samples"], model["bandwidth"])
                by_model[key] = {"indices": [], "z_grid": z_grid, "logpdf_grid": logpdf_grid}
            by_model[key]["indices"].append(gi)
        grouped[int(node)] = [
            (np.asarray(item["indices"], dtype=int), item["z_grid"], item["logpdf_grid"])
            for item in by_model.values()
        ]
    return grouped


def _load_artifacts(project_root: Path, prefix: str) -> dict[str, Any]:
    processed_dir = project_root / "data" / "processed"
    arrays = np.load(processed_dir / f"{prefix}_arrays.npz", allow_pickle=True)
    with open(processed_dir / f"{prefix}_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    examples = pd.read_csv(processed_dir / f"{prefix}_examples_index.csv")
    examples["datetime"] = pd.to_datetime(examples["datetime"])
    examples["subset_str"] = examples["subset_str"].astype(str)
    node_features = pd.read_csv(processed_dir / f"{prefix}_node_features.csv")
    node_features["datetime"] = pd.to_datetime(node_features["datetime"])
    node_features["node"] = node_features["node"].astype(int)
    node_gt = pd.read_csv(processed_dir / f"{prefix}_ground_truth_nodes.csv")
    node_gt["datetime"] = pd.to_datetime(node_gt["datetime"])
    node_gt["node"] = node_gt["node"].astype(int)
    vehicle_gt = pd.read_csv(processed_dir / f"{prefix}_ground_truth_vehicle.csv")
    vehicle_gt["datetime"] = pd.to_datetime(vehicle_gt["datetime"])
    sensor_geometry = pd.read_csv(processed_dir / f"{prefix}_sensor_geometry.csv")
    sensor_geometry["node"] = sensor_geometry["node"].astype(int)
    return {
        "processed_dir": processed_dir,
        "arrays": arrays,
        "meta": meta,
        "examples": examples,
        "node_features": node_features,
        "node_gt": node_gt,
        "vehicle_gt": vehicle_gt,
        "sensor_geometry": sensor_geometry,
    }


def score_rssi_two_tower(
    project_root: str | Path | None = None,
    *,
    prefix: str = DEFAULT_PREFIX,
    model_tag: str = DEFAULT_MODEL_TAG,
    device: str | None = None,
) -> pd.DataFrame:
    project_root = Path(project_root or Path.cwd())
    if project_root.name == "notebooks":
        project_root = project_root.parent
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = _load_artifacts(project_root, prefix)
    arrays = data["arrays"]
    meta = data["meta"]
    examples = data["examples"]
    sensor_geometry = data["sensor_geometry"]
    ordered_nodes = [int(x) for x in meta["ordered_nodes"]]
    if verbose:
        print("[runtime] Loading RSSI two-tower checkpoint and static action embeddings", flush=True)
    model_dir = project_root / "experiments" / model_tag
    table_dir = project_root / "reports" / "tables" / model_tag
    table_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"{model_tag}_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing {checkpoint_path}; run notebook 08 cell 2 first.")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TrainConfig(**checkpoint["config"])
    model = TwoTowerMLP(
        context_dim=int(checkpoint["context_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden=config.hidden,
        emb_dim=config.emb_dim,
        depth=config.depth,
        dropout=config.dropout,
        combine_mode=config.combine_mode,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    context_names = list(checkpoint["context_feature_names"])
    context_index = {name: i for i, name in enumerate(meta["context_feature_names"])}
    missing = [name for name in context_names if name not in context_index]
    if missing:
        raise KeyError(f"Missing checkpoint context features in processed data: {missing}")
    c_raw = np.column_stack([arrays["C_by_time"][:, context_index[name]] for name in context_names]).astype(np.float32)

    a_examples, _, _, action_names = _static_action_catalog_from_geometry(
        examples,
        sensor_geometry,
        ordered_nodes,
        max_subset_size=int(meta.get("max_subset_size", 3)),
    )
    if action_names != list(checkpoint["static_action_feature_names"]):
        raise RuntimeError("Action features do not match the trained RSSI-only checkpoint.")

    std = checkpoint["standardizers"]
    c_mu = np.asarray(std["C_mu"], dtype=np.float32)
    c_sigma = np.asarray(std["C_sigma"], dtype=np.float32)
    a_mu = np.asarray(std["A_mu"], dtype=np.float32)
    a_sigma = np.asarray(std["A_sigma"], dtype=np.float32)
    c_sigma = np.where(np.abs(c_sigma) < 1e-12, 1.0, c_sigma).astype(np.float32)
    a_sigma = np.where(np.abs(a_sigma) < 1e-12, 1.0, a_sigma).astype(np.float32)
    c_std = ((c_raw - c_mu) / c_sigma).astype(np.float32)
    a_std = ((a_examples - a_mu) / a_sigma).astype(np.float32)

    y_examples = arrays["y_examples"].astype(np.float32)
    time_id = arrays["example_time_id"].astype(np.int64)
    split = chronological_split(time_id, n_times=len(c_std), train_frac=config.train_frac, val_frac=config.val_frac)
    all_idx = np.arange(len(y_examples), dtype=np.int64)
    scores = predict_scores(model, c_std, a_std, time_id, all_idx, max(32768, config.batch_size * 2), device)

    split_by_time = {}
    for split_name in ("train", "val", "test"):
        for tid in split[f"{split_name}_time_ids"]:
            split_by_time[int(tid)] = split_name
    per_time = build_per_time_decisions(examples, y_examples, scores, split_by_time)
    per_time_path = table_dir / f"{model_tag}_per_time_decisions.csv"
    per_time.to_csv(per_time_path, index=False)
    return per_time


def _make_grid(sensor_xy: dict[int, tuple[float, float]], fit_nodes: list[int]) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    offsets = np.arange(-GRID_RADIUS_CELLS, GRID_RADIUS_CELLS)
    dxs, dys = np.meshgrid(offsets, offsets)
    center_x = float(np.mean([sensor_xy[n][0] for n in fit_nodes]))
    center_y = float(np.mean([sensor_xy[n][1] for n in fit_nodes]))
    grid_x = center_x + dxs.ravel() * GRID_SPACING_M
    grid_y = center_y + dys.ravel() * GRID_SPACING_M
    sensor_xy_arr = np.asarray([sensor_xy[n] for n in fit_nodes], dtype=float)
    nearest_nodes = []
    for gx, gy in zip(grid_x, grid_y):
        d = np.sqrt((sensor_xy_arr[:, 0] - gx) ** 2 + (sensor_xy_arr[:, 1] - gy) ** 2)
        nearest_nodes.append([fit_nodes[i] for i in np.argsort(d)[:SENSORS_PER_CELL]])
    return grid_x, grid_y, nearest_nodes


def _baseline_rows_from_grid(
    test_per: pd.DataFrame,
    rssi_wide: pd.DataFrame,
    fit_nodes: list[int],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    nearest_nodes: list[list[int]],
    loglike_fn,
    metric_col: str,
) -> pd.DataFrame:
    rows = []
    for row in test_per.itertuples(index=False):
        dt = pd.Timestamp(row.datetime)
        if dt not in rssi_wide.index:
            continue
        ll = np.zeros(len(grid_x), dtype=float)
        any_obs = False
        for node in fit_nodes:
            z = float(rssi_wide.at[dt, node]) if node in rssi_wide.columns else np.nan
            if not np.isfinite(z):
                continue
            any_obs = True
            ll += loglike_fn(node, z)
        if not any_obs:
            continue
        ll -= np.max(ll)
        posterior = np.exp(ll)
        s = float(posterior.sum())
        if s <= 0 or not np.isfinite(s):
            continue
        posterior /= s
        top_idx = np.argsort(posterior)[::-1][:TOP_K_CELLS]
        activated = set()
        for gi in top_idx:
            activated.update(nearest_nodes[int(gi)])
        true_closest = int(row.closest_node)
        rows.append(
            {
                "time_id": int(row.time_id),
                "datetime": dt,
                metric_col: int(true_closest in activated),
                "true_closest_node": true_closest,
                "activated_nodes": "-".join(map(str, sorted(activated))),
                "n_activated": int(len(activated)),
                "posterior_max": float(np.max(posterior)),
                "map_grid_x_m": float(grid_x[int(np.argmax(posterior))]),
                "map_grid_y_m": float(grid_y[int(np.argmax(posterior))]),
            }
        )
    return pd.DataFrame(rows).sort_values("time_id").reset_index(drop=True)


def run_comparison(
    project_root: str | Path | None = None,
    *,
    prefix: str = DEFAULT_PREFIX,
    model_tag: str = DEFAULT_MODEL_TAG,
    device: str | None = None,
    rolling_window: int = 100,
) -> dict[str, Any]:
    project_root = Path(project_root or Path.cwd())
    if project_root.name == "notebooks":
        project_root = project_root.parent
    table_dir = project_root / "reports" / "tables" / model_tag
    fig_dir = project_root / "reports" / "figures" / model_tag
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_time = score_rssi_two_tower(project_root, prefix=prefix, model_tag=model_tag, device=device)
    data = _load_artifacts(project_root, prefix)
    node_features = data["node_features"]
    node_gt = data["node_gt"]
    sensor_geometry = data["sensor_geometry"]
    ordered_nodes = [int(x) for x in data["meta"]["ordered_nodes"]]

    obs = (
        node_features[["datetime", "node", "rssi_db"]]
        .merge(node_gt[["datetime", "node", "distance_to_vehicle_m"]], on=["datetime", "node"], how="inner")
        .dropna(subset=["rssi_db", "distance_to_vehicle_m"])
    )
    train_times = set(per_time.loc[per_time["split"].eq("train"), "datetime"])
    train_obs = obs[obs["datetime"].isin(train_times)].copy()
    test_per = per_time.loc[per_time["split"].eq("test")].sort_values("time_id").copy()
    if len(test_per) == 0:
        raise ValueError("No test rows found for Grace's Quarters comparison.")

    sensor_xy = {
        int(row.node): (float(row.sensor_x_m), float(row.sensor_y_m))
        for row in sensor_geometry.itertuples(index=False)
    }
    rssi_wide = node_features.pivot_table(index="datetime", columns="node", values="rssi_db", aggfunc="first").sort_index()

    print(f"train RSSI-distance rows: {len(train_obs):,}")
    print(f"test timesteps: {len(test_per):,}")
    print(f"nodes: {ordered_nodes}")

    pathloss = _fit_pathloss(train_obs, ordered_nodes)
    fit_nodes = [n for n in ordered_nodes if n in pathloss]
    if not fit_nodes:
        raise ValueError("No path-loss models fit.")
    grid_x, grid_y, nearest_nodes = _make_grid(sensor_xy, fit_nodes)

    # Normalized top-p baseline.
    norm_rows = []
    for row in test_per.itertuples(index=False):
        dt = pd.Timestamp(row.datetime)
        if dt not in rssi_wide.index:
            continue
        scores = {}
        for node in fit_nodes:
            if node not in rssi_wide.columns:
                continue
            z = float(rssi_wide.at[dt, node])
            if not np.isfinite(z) or abs(pathloss[node]["eta"]) < EPS:
                continue
            scores[node] = float((z - pathloss[node]["P0"]) / pathloss[node]["eta"])
        if not scores:
            continue
        selected = [node for node, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:OUTPUT_SET_SIZE]]
        true_closest = int(row.closest_node)
        norm_rows.append(
            {
                "time_id": int(row.time_id),
                "datetime": pd.Timestamp(row.datetime),
                "normalized_top_p_success": int(true_closest in selected),
                "normalized_contains_closest": int(true_closest in selected),
                "selected_nodes": "-".join(map(str, selected)),
                "true_closest_node": true_closest,
                "output_set_size": len(selected),
            }
        )
    norm_df = pd.DataFrame(norm_rows).sort_values("time_id").reset_index(drop=True)
    norm_df.to_csv(table_dir / "graces_normalized_top_p_baseline_per_time_test.csv", index=False)

    # Linear path-loss posterior.
    pred_grid = {}
    sigma_grid = {}
    for node in fit_nodes:
        sx, sy = sensor_xy[node]
        d_grid = np.sqrt((grid_x - sx) ** 2 + (grid_y - sy) ** 2) + EPS
        pred_grid[node] = pathloss[node]["P0"] - pathloss[node]["eta"] * _logdist(d_grid)
        sigma_grid[node] = pathloss[node]["sigma2"]

    linear_df = _baseline_rows_from_grid(
        test_per,
        rssi_wide,
        fit_nodes,
        grid_x,
        grid_y,
        nearest_nodes,
        lambda node, z: _gaussian_logpdf(z, pred_grid[node], sigma_grid[node]),
        "pathloss_contains_closest",
    )
    linear_df.to_csv(table_dir / "graces_pathloss_baseline_per_time_test.csv", index=False)

    # CVXPY spline posterior.
    spline = _fit_spline(train_obs, ordered_nodes)
    spline_nodes = [n for n in ordered_nodes if n in spline]
    if not spline_nodes:
        raise ValueError("No spline models fit.")
    spline_grid_x, spline_grid_y, spline_nearest = _make_grid(sensor_xy, spline_nodes)
    spline_mu = {}
    spline_sigma2 = {}
    for node in spline_nodes:
        sx, sy = sensor_xy[node]
        d_grid = np.sqrt((spline_grid_x - sx) ** 2 + (spline_grid_y - sy) ** 2) + EPS
        spline_mu[node], spline_sigma2[node] = _spline_predict(spline[node], d_grid)
    spline_df = _baseline_rows_from_grid(
        test_per,
        rssi_wide,
        spline_nodes,
        spline_grid_x,
        spline_grid_y,
        spline_nearest,
        lambda node, z: _gaussian_logpdf(z, spline_mu[node], spline_sigma2[node]),
        "spline_contains_closest",
    )
    spline_df.to_csv(table_dir / "graces_spline_cvxpy_baseline_per_time_test.csv", index=False)

    # KDE-hybrid posterior with path-loss fallback.
    kde = _fit_kde(train_obs, ordered_nodes)
    kde_nodes = [n for n in ordered_nodes if n in pathloss]
    kde_grid_x, kde_grid_y, kde_nearest = _make_grid(sensor_xy, kde_nodes)
    kde_alloc = {}
    fallback_mu = {}
    for node in kde_nodes:
        sx, sy = sensor_xy[node]
        d_grid = np.sqrt((kde_grid_x - sx) ** 2 + (kde_grid_y - sy) ** 2) + EPS
        fallback_mu[node] = pathloss[node]["P0"] - pathloss[node]["eta"] * _logdist(d_grid)
        alloc = []
        bins = kde.get(node, [])
        centers = np.asarray([b["center"] for b in bins], dtype=float)
        for d in d_grid:
            if len(centers):
                k = int(np.argmin(np.abs(centers - d)))
                if abs(float(d) - float(centers[k])) <= KDE_SUPPORT_DELTA_M:
                    alloc.append(bins[k])
                    continue
            alloc.append(None)
        kde_alloc[node] = alloc

    def kde_ll(node: int, z: float) -> np.ndarray:
        out = np.empty(len(kde_grid_x), dtype=float)
        for gi, model in enumerate(kde_alloc[node]):
            if model is None:
                out[gi] = _gaussian_logpdf(z, fallback_mu[node][gi], pathloss[node]["sigma2"])
            else:
                out[gi] = _kde_logpdf(z, model["samples"], model["bandwidth"])
        return out

    kde_df = _baseline_rows_from_grid(
        test_per,
        rssi_wide,
        kde_nodes,
        kde_grid_x,
        kde_grid_y,
        kde_nearest,
        kde_ll,
        "kde_hybrid_contains_closest",
    )
    kde_df.to_csv(table_dir / "graces_kde_hybrid_baseline_per_time_test.csv", index=False)

    method_specs = [
        ("RSSI Two-Tower", per_time.loc[per_time["split"].eq("test")], "contains_closest"),
        ("Linear fit", linear_df, "pathloss_contains_closest"),
        ("Spline fit", spline_df, "spline_contains_closest"),
        ("KDE", kde_df, "kde_hybrid_contains_closest"),
        ("Normalized", norm_df, "normalized_contains_closest"),
    ]

    merged = None
    for method, df, metric in method_specs:
        tmp = df[["time_id", "datetime", metric]].rename(columns={metric: method}).copy()
        tmp["datetime"] = pd.to_datetime(tmp["datetime"])
        if merged is None:
            merged = tmp
        else:
            merged = merged.merge(tmp.drop(columns=["datetime"]), on="time_id", how="outer")
    assert merged is not None
    merged = merged.sort_values("time_id").reset_index(drop=True)
    methods = [m for m, _, _ in method_specs]

    summary_rows = []
    for method in methods:
        vals = pd.to_numeric(merged[method], errors="coerce").dropna()
        mean = float(vals.mean())
        n = int(len(vals))
        summary_rows.append(
            {
                "method": method,
                "n_valid": n,
                "mean_accuracy": mean,
                "std_error_binomial": float(np.sqrt(mean * (1.0 - mean) / n)) if n else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
    merged.to_csv(table_dir / "graces_all_methods_merged_per_time_accuracy.csv", index=False)
    summary.to_csv(table_dir / "graces_all_methods_mean_accuracy_table.csv", index=False)

    # Compact time-vs-accuracy figure, matching the notebook-07 report style.
    t0 = pd.to_datetime(merged["datetime"]).dropna().iloc[0]
    elapsed_s = (pd.to_datetime(merged["datetime"]) - t0).dt.total_seconds()
    style = {
        "RSSI Two-Tower": dict(color="tab:purple", linewidth=3.0, linestyle="-"),
        "Linear fit": dict(color="tab:blue", linewidth=2.2, linestyle="--"),
        "Spline fit": dict(color="tab:orange", linewidth=2.2, linestyle="-."),
        "KDE": dict(color="tab:green", linewidth=2.4, linestyle=":"),
        "Normalized": dict(color="tab:red", linewidth=2.2, linestyle=(0, (5.0, 1.5, 1.2, 1.5))),
    }
    fig, ax = plt.subplots(figsize=(7.35, 3.55))
    for method in methods:
        roll = pd.to_numeric(merged[method], errors="coerce").rolling(rolling_window, min_periods=1).mean()
        ax.plot(elapsed_s, roll, label=method, **style[method])
    ax.set_ylim(0.5, 1.03)
    ax.set_xlabel("Elapsed time in test split (s)")
    ax.set_ylabel("Rolling contains-closest accuracy")
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "graces_all_methods_time_vs_accuracy.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "graces_all_methods_time_vs_accuracy.png", dpi=300, bbox_inches="tight")

    print("Saved Grace's Quarters comparison outputs:")
    print(" ", table_dir / "graces_all_methods_mean_accuracy_table.csv")
    print(" ", table_dir / "graces_all_methods_merged_per_time_accuracy.csv")
    print(" ", fig_dir / "graces_all_methods_time_vs_accuracy.pdf")

    return {
        "per_time": per_time,
        "linear": linear_df,
        "spline": spline_df,
        "kde": kde_df,
        "normalized": norm_df,
        "merged": merged,
        "summary": summary,
        "paths": {
            "table_dir": table_dir,
            "fig_dir": fig_dir,
        },
    }


def _load_method_csv(label: str, path: Path, metric_candidates: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label}: missing {path}. Run the comparison cell first.")
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower().eq("test")].copy()
    metric = next((col for col in metric_candidates if col in df.columns), None)
    if metric is None:
        raise ValueError(f"{label}: none of {metric_candidates} in {path.name}; columns={list(df.columns)}")
    if "time_id" not in df.columns:
        df["time_id"] = np.arange(len(df), dtype=int)
    keep = ["time_id", metric]
    if "datetime" in df.columns:
        keep.insert(1, "datetime")
    out = df[keep].copy().rename(columns={metric: label})
    out["time_id"] = out["time_id"].astype(int)
    out[label] = pd.to_numeric(out[label], errors="coerce")
    agg = {label: "mean"}
    if "datetime" in out.columns:
        agg["datetime"] = "first"
    return out.groupby("time_id", as_index=False).agg(agg)


def _accuracy_ylim(curves, pad: float = 0.025) -> tuple[float, float]:
    hi = 1.005

    vals = []
    for curve in curves:
        arr = np.asarray(curve, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            vals.append(arr)

    if not vals:
        return 0.5, hi

    y = np.concatenate(vals)
    lo = max(0.5, float(np.min(y)) - pad)

    return lo, hi


def _runtime_summary(values_ms: np.ndarray, n_steps: int, repeats: int) -> dict[str, Any]:
    values_ms = np.asarray(values_ms, dtype=float)
    return {
        "mean_ms": float(values_ms.mean()),
        "median_ms": float(np.median(values_ms)),
        "p95_ms": float(np.percentile(values_ms, 95)),
        "p99_ms": float(np.percentile(values_ms, 99)),
        "std_ms": float(values_ms.std()),
        "min_ms": float(values_ms.min()),
        "max_ms": float(values_ms.max()),
        "n_timed_calls": int(values_ms.size),
        "n_test_steps": int(n_steps),
        "n_repeats": int(repeats),
    }


def _benchmark_component(
    name: str,
    time_ids: np.ndarray,
    fn,
    *,
    repeats: int,
    warmup: int,
    verbose: bool = False,
    progress_every: int = 500,
) -> tuple[dict[str, Any], pd.DataFrame]:
    n_steps = len(time_ids)
    progress_every = max(1, int(progress_every))
    if verbose:
        print(f"[runtime] {name}: warmup {warmup} steps, then {repeats} x {n_steps} timed calls", flush=True)
    warmup_t0 = time.perf_counter()
    for tid in time_ids[:warmup]:
        fn(int(tid))
    if verbose:
        print(f"[runtime] {name}: warmup done in {time.perf_counter() - warmup_t0:.1f}s", flush=True)
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    rows = []
    timed_t0 = time.perf_counter()
    try:
        for repeat in range(repeats):
            repeat_t0 = time.perf_counter()
            for step_idx, tid in enumerate(time_ids):
                t0 = time.perf_counter_ns()
                fn(int(tid))
                t1 = time.perf_counter_ns()
                rows.append((name, repeat, step_idx, int(tid), (t1 - t0) / 1e6))
                if verbose and ((step_idx + 1) % progress_every == 0 or (step_idx + 1) == n_steps):
                    done = repeat * n_steps + step_idx + 1
                    total = repeats * n_steps
                    elapsed = time.perf_counter() - timed_t0
                    rate = done / max(elapsed, 1e-9)
                    eta = (total - done) / max(rate, 1e-9)
                    print(
                        f"[runtime] {name}: repeat {repeat + 1}/{repeats}, "
                        f"step {step_idx + 1}/{n_steps}, {100.0 * done / total:.1f}% "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
            if verbose:
                print(f"[runtime] {name}: repeat {repeat + 1}/{repeats} done in {time.perf_counter() - repeat_t0:.1f}s", flush=True)
    finally:
        if was_enabled:
            gc.enable()
    raw = pd.DataFrame(rows, columns=["component", "repeat", "step_idx", "time_id", "runtime_ms"])
    summary = {"component": name, **_runtime_summary(raw["runtime_ms"].to_numpy(dtype=float), len(time_ids), repeats)}
    if verbose:
        print(
            f"[runtime] {name}: mean={summary['mean_ms']:.4f} ms, "
            f"p95={summary['p95_ms']:.4f} ms, p99={summary['p99_ms']:.4f} ms",
            flush=True,
        )
    return summary, raw


def make_report_plot_and_runtime(
    project_root: str | Path | None = None,
    *,
    prefix: str = DEFAULT_PREFIX,
    model_tag: str = DEFAULT_MODEL_TAG,
    repeats: int = 20,
    warmup: int = 200,
    rolling_window: int = 100,
    verbose: bool = False,
    runtime_progress_every: int = 500,
) -> dict[str, Any]:
    wall_t0 = time.perf_counter()
    project_root = Path(project_root or Path.cwd())
    if project_root.name == "notebooks":
        project_root = project_root.parent
    table_dir = project_root / "reports" / "tables" / model_tag
    fig_dir = project_root / "reports" / "figures" / model_tag
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_per_time_path = table_dir / f"{model_tag}_per_time_decisions.csv"
    paths = {
        "RSSI Two-Tower": model_per_time_path,
        "Linear fit": table_dir / "graces_pathloss_baseline_per_time_test.csv",
        "Spline fit": table_dir / "graces_spline_cvxpy_baseline_per_time_test.csv",
        "KDE": table_dir / "graces_kde_hybrid_baseline_per_time_test.csv",
        "Normalized": table_dir / "graces_normalized_top_p_baseline_per_time_test.csv",
    }
    if verbose:
        print(f"[report] Loading existing per-time method outputs from {table_dir}", flush=True)
    method_dfs = [
        _load_method_csv("RSSI Two-Tower", paths["RSSI Two-Tower"], ["contains_closest"]),
        _load_method_csv("Linear fit", paths["Linear fit"], ["pathloss_contains_closest"]),
        _load_method_csv("Spline fit", paths["Spline fit"], ["spline_contains_closest"]),
        _load_method_csv("KDE", paths["KDE"], ["kde_hybrid_contains_closest"]),
        _load_method_csv("Normalized", paths["Normalized"], ["normalized_contains_closest", "normalized_top_p_success"]),
    ]

    merged = None
    for df in method_dfs:
        if merged is None:
            merged = df.copy()
        else:
            if "datetime" in merged.columns and "datetime" in df.columns:
                df = df.drop(columns=["datetime"])
            merged = merged.merge(df, on="time_id", how="outer")
    assert merged is not None
    merged = merged.sort_values("time_id").reset_index(drop=True)

    methods = ["RSSI Two-Tower", "Linear fit", "Spline fit", "KDE", "Normalized"]
    common_mask = merged[methods].notna().all(axis=1)
    summary_rows = []
    display_names = {
        "RSSI Two-Tower": "Two-Tower (RSSI)",
        "Linear fit": "Linear",
        "Spline fit": "Spline",
        "KDE": "KDE",
        "Normalized": "Norm.",
    }
    for method in methods:
        vals = pd.to_numeric(merged[method], errors="coerce").dropna()
        common_vals = pd.to_numeric(merged.loc[common_mask, method], errors="coerce").dropna()
        mean = float(vals.mean())
        n = int(vals.shape[0])
        summary_rows.append(
            {
                "method": method,
                "display_name": display_names[method],
                "n_valid": n,
                "mean_accuracy": mean,
                "std_error_binomial": float(np.sqrt(mean * (1.0 - mean) / n)) if n else np.nan,
                "n_common_times": int(common_vals.shape[0]),
                "mean_accuracy_common_times": float(common_vals.mean()) if len(common_vals) else np.nan,
                "rolling_window": rolling_window,
                "source_csv": str(paths[method]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    if verbose:
        print("[report] Mean accuracies:", flush=True)
        for row in summary.itertuples(index=False):
            print(f"  {row.display_name}: {100.0 * row.mean_accuracy:.2f}% over {row.n_valid} timestamps", flush=True)

    styled_summary_path = table_dir / "graces_all_methods_report_mean_accuracy_table.csv"
    styled_merged_path = table_dir / "graces_all_methods_report_merged_per_time_accuracy.csv"
    latex_path = table_dir / "graces_all_methods_report_mean_accuracy_table_latex.txt"
    summary.to_csv(styled_summary_path, index=False)
    merged.to_csv(styled_merged_path, index=False)
    latex_path.write_text(
        "\n".join(
            f"{row.display_name} & {100.0 * row.mean_accuracy:.2f} & {100.0 * row.std_error_binomial:.2f} \\\\"
            for row in summary.itertuples(index=False)
        ),
        encoding="utf-8",
    )

    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.15,
        "xtick.major.width": 1.05,
        "ytick.major.width": 1.05,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
    })
    style = {
        "RSSI Two-Tower": {"label": "Two-Tower (RSSI)", "color": "black", "linestyle": "-", "linewidth": 5.0, "marker": None, "markersize": 0, "markevery": None, "zorder": 40, "alpha": 1.0},
        "Linear fit": {"label": "Linear", "color": "tab:blue", "linestyle": "--", "linewidth": 3.6, "marker": "s", "markersize": 4.0, "markevery": 155, "zorder": 10, "alpha": 0.95},
        "Spline fit": {"label": "Spline", "color": "tab:orange", "linestyle": "-.", "linewidth": 3.6, "marker": "^", "markersize": 4.4, "markevery": 185, "zorder": 10, "alpha": 0.95},
        "KDE": {"label": "KDE", "color": "tab:green", "linestyle": ":", "linewidth": 3.8, "marker": "D", "markersize": 4.0, "markevery": 215, "zorder": 10, "alpha": 0.95},
        "Normalized": {"label": "Norm.", "color": "tab:red", "linestyle": (0, (5.0, 1.5, 1.2, 1.5)), "linewidth": 3.6, "marker": "x", "markersize": 5.0, "markevery": 245, "zorder": 10, "alpha": 0.95},
    }
    if "datetime" in merged.columns and merged["datetime"].notna().any():
        dt = pd.to_datetime(merged["datetime"], errors="coerce")
        x = (dt - dt.dropna().iloc[0]).dt.total_seconds()
        x_label = "Time (s)"
    else:
        x = merged["time_id"]
        x_label = "Test index"

    fig, ax = plt.subplots(figsize=(7.35, 3.55), constrained_layout=True)
    for method in methods:
        y = pd.to_numeric(merged[method], errors="coerce").rolling(rolling_window, min_periods=1).mean()
        st = style[method]
        kwargs = dict(label=st["label"], color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"], zorder=st["zorder"], alpha=st["alpha"])
        if st["marker"] is not None:
            kwargs.update(marker=st["marker"], markersize=st["markersize"], markevery=st["markevery"])
        ax.plot(x, y, **kwargs)
    ax.set_ylim(0.5, 1.03)
    ax.set_xlabel(x_label, labelpad=4)
    ax.set_ylabel("Accuracy", labelpad=5)
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.32)
    ax.set_axisbelow(True)
    ax.margins(x=0.01)
    leg = ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.25", borderpad=0.32, handlelength=2.05, handletextpad=0.48, columnspacing=0.75, labelspacing=0.20)
    leg.set_zorder(10000)
    for text in leg.get_texts():
        if text.get_text() == "Two-Tower (RSSI)":
            text.set_fontweight("bold")
    plot_pdf = fig_dir / "graces_all_methods_report_time_vs_accuracy_double_column_styled.pdf"
    plot_png = fig_dir / "graces_all_methods_report_time_vs_accuracy_double_column_styled.png"
    fig.savefig(plot_pdf, bbox_inches="tight")
    fig.savefig(plot_png, dpi=700, bbox_inches="tight")
    if verbose:
        print(f"[report] Saved accuracy plot to {plot_pdf}", flush=True)

    # Runtime benchmarking.
    if verbose:
        print("[runtime] Loading processed arrays and model-side metadata", flush=True)
    data = _load_artifacts(project_root, prefix)
    arrays = data["arrays"]
    meta = data["meta"]
    node_features = data["node_features"]
    node_gt = data["node_gt"]
    sensor_geometry = data["sensor_geometry"]
    ordered_nodes = [int(x) for x in meta["ordered_nodes"]]
    base_context_index = {name: i for i, name in enumerate(meta["context_feature_names"])}
    per_time = pd.read_csv(model_per_time_path)
    test_time_ids = per_time.loc[per_time["split"].astype(str).eq("test"), "time_id"].astype(int).to_numpy()
    if len(test_time_ids) == 0:
        raise ValueError("No test time ids found for runtime benchmark.")
    warmup = min(warmup, len(test_time_ids))
    repeats = int(repeats)
    if verbose:
        print(
            f"[runtime] Benchmark setup: {len(test_time_ids)} test timestamps, "
            f"{repeats} repeats, {warmup} warmup steps, {len(ordered_nodes)} nodes",
            flush=True,
        )

    rssi_by_time = np.column_stack([arrays["C_by_time"][:, base_context_index[f"n{node}_rssi_db"]] for node in ordered_nodes]).astype(np.float32)
    sensor_x = np.array([arrays["C_by_time"][0, base_context_index[f"n{node}_sensor_x_norm"]] for node in ordered_nodes], dtype=np.float32)
    sensor_y = np.array([arrays["C_by_time"][0, base_context_index[f"n{node}_sensor_y_norm"]] for node in ordered_nodes], dtype=np.float32)

    def extract_rssi_scalar_features(tid: int) -> np.ndarray:
        return rssi_by_time[int(tid)].copy()

    feature_summary, feature_raw = _benchmark_component(
        "RSSI-only scalar extraction (10 sensors)",
        test_time_ids,
        extract_rssi_scalar_features,
        repeats=repeats,
        warmup=warmup,
        verbose=verbose,
        progress_every=runtime_progress_every,
    )
    rssi_cache = {int(tid): extract_rssi_scalar_features(int(tid)) for tid in test_time_ids}

    # Fit baseline models outside the timed loop.
    if verbose:
        print("[runtime] Fitting path-loss, spline, and KDE baselines outside timed loops", flush=True)
    obs = (
        node_features[["datetime", "node", "rssi_db"]]
        .merge(node_gt[["datetime", "node", "distance_to_vehicle_m"]], on=["datetime", "node"], how="inner")
        .dropna(subset=["rssi_db", "distance_to_vehicle_m"])
    )
    per_time_full = pd.read_csv(model_per_time_path, parse_dates=["datetime"])
    train_times = set(per_time_full.loc[per_time_full["split"].astype(str).eq("train"), "datetime"])
    train_obs = obs[obs["datetime"].isin(train_times)].copy()
    sensor_xy = {
        int(row.node): (float(row.sensor_x_m), float(row.sensor_y_m))
        for row in sensor_geometry.itertuples(index=False)
    }
    node_pos = {node: i for i, node in enumerate(ordered_nodes)}
    if verbose:
        print("[runtime] Fitting linear path-loss models", flush=True)
    pathloss = _fit_pathloss(train_obs, ordered_nodes)
    fit_nodes = [n for n in ordered_nodes if n in pathloss]
    grid_x, grid_y, nearest_nodes = _make_grid(sensor_xy, fit_nodes)
    if verbose:
        print(f"[runtime] Path-loss fit nodes={len(fit_nodes)}, grid points={len(grid_x)}", flush=True)
    pred_grid = {}
    for node in fit_nodes:
        sx, sy = sensor_xy[node]
        dist_grid = np.sqrt((grid_x - sx) ** 2 + (grid_y - sy) ** 2) + EPS
        pred_grid[node] = pathloss[node]["P0"] - pathloss[node]["eta"] * _logdist(dist_grid)

    if verbose:
        print("[runtime] Fitting CVXPY spline models", flush=True)
    spline = _fit_spline(train_obs, ordered_nodes)
    spline_nodes = [n for n in ordered_nodes if n in spline]
    spline_grid_x, spline_grid_y, spline_nearest = _make_grid(sensor_xy, spline_nodes)
    if verbose:
        print(f"[runtime] Spline fit nodes={len(spline_nodes)}, grid points={len(spline_grid_x)}", flush=True)
    spline_mu = {}
    spline_sigma2 = {}
    for node in spline_nodes:
        sx, sy = sensor_xy[node]
        dist_grid = np.sqrt((spline_grid_x - sx) ** 2 + (spline_grid_y - sy) ** 2) + EPS
        spline_mu[node], spline_sigma2[node] = _spline_predict(spline[node], dist_grid)

    if verbose:
        print("[runtime] Fitting KDE-hybrid models", flush=True)
    kde = _fit_kde(train_obs, ordered_nodes)
    kde_nodes = [n for n in ordered_nodes if n in pathloss]
    kde_grid_x, kde_grid_y, kde_nearest = _make_grid(sensor_xy, kde_nodes)
    if verbose:
        n_kde_bins = sum(len(v) for v in kde.values())
        print(f"[runtime] KDE fit nodes={len(kde_nodes)}, bins={n_kde_bins}, grid points={len(kde_grid_x)}", flush=True)
    kde_fallback_mu = {}
    kde_alloc = {}
    for node in kde_nodes:
        sx, sy = sensor_xy[node]
        dist_grid = np.sqrt((kde_grid_x - sx) ** 2 + (kde_grid_y - sy) ** 2) + EPS
        kde_fallback_mu[node] = pathloss[node]["P0"] - pathloss[node]["eta"] * _logdist(dist_grid)
        bins = kde.get(node, [])
        centers = np.asarray([b["center"] for b in bins], dtype=float)
        alloc = []
        for d in dist_grid:
            if len(centers):
                k = int(np.argmin(np.abs(centers - d)))
                if abs(float(d) - float(centers[k])) <= KDE_SUPPORT_DELTA_M:
                    alloc.append(bins[k])
                    continue
            alloc.append(None)
        kde_alloc[node] = alloc
    if verbose:
        print(
            f"[runtime] Precomputing KDE lookup tables with {KDE_LOOKUP_GRID_SIZE} RSSI grid points",
            flush=True,
        )
    kde_lookup_t0 = time.perf_counter()
    kde_lookup_groups = _make_kde_lookup_groups(kde_alloc)
    if verbose:
        n_lookup_groups = sum(len(groups) for groups in kde_lookup_groups.values())
        print(
            f"[runtime] KDE lookup groups={n_lookup_groups}, precompute={time.perf_counter() - kde_lookup_t0:.1f}s",
            flush=True,
        )

    def decide_normalized(tid: int) -> tuple[int, ...]:
        rssi = rssi_cache[int(tid)]
        scores = []
        for node in fit_nodes:
            eta = pathloss[node]["eta"]
            if abs(eta) < EPS:
                continue
            score = (float(rssi[node_pos[node]]) - pathloss[node]["P0"]) / eta
            scores.append((node, score))
        return tuple(node for node, _ in sorted(scores, key=lambda kv: kv[1], reverse=True)[:OUTPUT_SET_SIZE])

    def decide_linear(tid: int) -> tuple[int, ...]:
        rssi = rssi_cache[int(tid)]
        ll = np.zeros(len(grid_x), dtype=float)
        for node in fit_nodes:
            ll += _gaussian_logpdf(float(rssi[node_pos[node]]), pred_grid[node], pathloss[node]["sigma2"])
        gi = int(np.argmax(ll))
        return tuple(nearest_nodes[gi])

    def decide_spline(tid: int) -> tuple[int, ...]:
        rssi = rssi_cache[int(tid)]
        ll = np.zeros(len(spline_grid_x), dtype=float)
        for node in spline_nodes:
            ll += _gaussian_logpdf(float(rssi[node_pos[node]]), spline_mu[node], spline_sigma2[node])
        gi = int(np.argmax(ll))
        return tuple(spline_nearest[gi])

    def decide_kde(tid: int) -> tuple[int, ...]:
        rssi = rssi_cache[int(tid)]
        ll = np.zeros(len(kde_grid_x), dtype=float)
        for node in kde_nodes:
            z = float(rssi[node_pos[node]])
            vals = _gaussian_logpdf(z, kde_fallback_mu[node], pathloss[node]["sigma2"]).astype(float)
            for indices, z_grid, logpdf_grid in kde_lookup_groups.get(node, []):
                vals[indices] = np.interp(z, z_grid, logpdf_grid)
            ll += vals
        gi = int(np.argmax(ll))
        return tuple(kde_nearest[gi])

    model_dir = project_root / "experiments" / model_tag
    ckpt_path = model_dir / f"{model_tag}_checkpoint.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    tower_model = TwoTowerMLP(
        context_dim=int(ckpt["context_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden=cfg.hidden,
        emb_dim=cfg.emb_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
        combine_mode=cfg.combine_mode,
    ).cpu()
    tower_model.load_state_dict(ckpt["model_state_dict"])
    tower_model.eval()
    emb = np.load(model_dir / "static_action_embeddings.npz", allow_pickle=True)["action_embeddings"].astype(np.float32)
    emb_t = torch.from_numpy(emb).cpu()
    if verbose:
        print(f"[runtime] Two-tower static actions={emb_t.shape[0]}, emb_dim={emb_t.shape[1]}", flush=True)
    c_mu = np.asarray(ckpt["standardizers"]["C_mu"], dtype=np.float32)
    c_sigma = np.asarray(ckpt["standardizers"]["C_sigma"], dtype=np.float32)
    c_sigma = np.where(np.abs(c_sigma) < 1e-12, 1.0, c_sigma).astype(np.float32)

    def assemble_tower_context(tid: int) -> np.ndarray:
        vals = []
        rssi = rssi_cache[int(tid)]
        for j, _node in enumerate(ordered_nodes):
            vals.extend([sensor_x[j], sensor_y[j], rssi[j]])
        raw = np.asarray(vals, dtype=np.float32)
        return ((raw - c_mu) / c_sigma).astype(np.float32)

    context_cache = {int(tid): assemble_tower_context(int(tid)) for tid in test_time_ids}

    @torch.no_grad()
    def decide_tower(tid: int) -> int:
        c = torch.from_numpy(context_cache[int(tid)][None, :]).cpu()
        c_emb = tower_model.embed_context(c).expand(emb_t.shape[0], -1)
        scores = tower_model.score_embeddings(c_emb, emb_t)
        return int(torch.argmax(scores).item())

    component_rows = []
    raw_rows = []
    try:
        context_summary, context_raw = _benchmark_component(
            "RSSI Two-Tower context/state assembly",
            test_time_ids,
            assemble_tower_context,
            repeats=repeats,
            warmup=warmup,
            verbose=verbose,
            progress_every=runtime_progress_every,
        )
        component_rows.append(context_summary)
        raw_rows.append(context_raw)

        for name, fn in [
            ("Normalized decision", decide_normalized),
            ("Linear fit decision", decide_linear),
            ("Spline fit decision", decide_spline),
            ("KDE decision", decide_kde),
            ("RSSI Two-Tower decision over static-action embeddings", decide_tower),
        ]:
            summary_i, raw_i = _benchmark_component(
                name,
                test_time_ids,
                fn,
                repeats=repeats,
                warmup=warmup,
                verbose=verbose,
                progress_every=runtime_progress_every,
            )
            component_rows.append(summary_i)
            raw_rows.append(raw_i)
    finally:
        torch.set_num_threads(old_threads)

    component_df = pd.DataFrame([feature_summary] + component_rows)
    raw_df = pd.concat([feature_raw] + raw_rows, ignore_index=True)
    component_df.to_csv(table_dir / "graces_all_methods_runtime_component_summary.csv", index=False)
    raw_df.to_csv(table_dir / "graces_all_methods_runtime_component_raw.csv", index=False)

    comp = component_df.set_index("component")
    feature_mean = float(comp.loc["RSSI-only scalar extraction (10 sensors)", "mean_ms"])
    feature_p95 = float(comp.loc["RSSI-only scalar extraction (10 sensors)", "p95_ms"])
    decision_map = {
        "Normalized": "Normalized decision",
        "Linear fit": "Linear fit decision",
        "Spline fit": "Spline fit decision",
        "KDE": "KDE decision",
        "RSSI Two-Tower": "RSSI Two-Tower decision over static-action embeddings",
    }
    context_mean_map = {method: 0.0 for method in methods}
    context_p95_map = {method: 0.0 for method in methods}
    context_mean_map["RSSI Two-Tower"] = float(comp.loc["RSSI Two-Tower context/state assembly", "mean_ms"])
    context_p95_map["RSSI Two-Tower"] = float(comp.loc["RSSI Two-Tower context/state assembly", "p95_ms"])

    acc_lookup = summary.set_index("method")["mean_accuracy"].to_dict()
    runtime_rows = []
    for method in ["Normalized", "Linear fit", "Spline fit", "RSSI Two-Tower", "KDE"]:
        dname = decision_map[method]
        decision_mean = float(comp.loc[dname, "mean_ms"])
        decision_p95 = float(comp.loc[dname, "p95_ms"])
        total_mean = feature_mean + context_mean_map[method] + decision_mean
        total_p95 = feature_p95 + context_p95_map[method] + decision_p95
        runtime_rows.append(
            {
                "method": method,
                "mean_accuracy": float(acc_lookup.get(method, np.nan)),
                "feature_profile": "RSSI-only scalar extraction (10 sensors)",
                "feature_mean_ms": feature_mean,
                "context_state_mean_ms": context_mean_map[method],
                "decision_mean_ms": decision_mean,
                "total_compute_mean_ms": total_mean,
                "total_compute_p95_component_sum_ms": total_p95,
                "total_compute_steps_per_second_mean": float(1000.0 / max(total_mean, 1e-12)),
                "included_compute": "RSSI feature extraction/readout + streaming state/context assembly where required + decision",
                "excluded": "sensor acquisition, network/MQTT, synchronization wait, packet loss handling, FLAC decode/file I/O, offline fitting/training",
            }
        )
    runtime = pd.DataFrame(runtime_rows)
    runtime_full_path = table_dir / "graces_all_methods_accuracy_and_compute_side_runtime.csv"
    runtime_paper_path = table_dir / "graces_all_methods_total_online_runtime_with_features.csv"
    runtime_latex_path = table_dir / "graces_all_methods_compute_side_runtime_latex.txt"
    runtime.to_csv(runtime_full_path, index=False)
    paper_runtime = runtime[[
        "method",
        "mean_accuracy",
        "feature_mean_ms",
        "context_state_mean_ms",
        "decision_mean_ms",
        "total_compute_mean_ms",
        "total_compute_p95_component_sum_ms",
    ]].copy()
    paper_runtime.to_csv(runtime_paper_path, index=False)
    runtime_latex_path.write_text(
        "\n".join(
            (
                f"\\textbf{{RSSI Two-Tower}} & {100.0 * row.mean_accuracy:.2f} & \\textbf{{{row.feature_mean_ms:.4f}}} & \\textbf{{{row.total_compute_mean_ms:.4f}}} & \\textbf{{{row.total_compute_p95_component_sum_ms:.4f}}} \\\\"
                if row.method == "RSSI Two-Tower"
                else f"{row.method} & {100.0 * row.mean_accuracy:.2f} & {row.feature_mean_ms:.4f} & {row.total_compute_mean_ms:.4f} & {row.total_compute_p95_component_sum_ms:.4f} \\\\"
            )
            for row in paper_runtime.itertuples(index=False)
        ),
        encoding="utf-8",
    )

    print("Saved Grace's Quarters report plot:", plot_pdf)
    print("Saved Grace's Quarters accuracy table:", styled_summary_path)
    print("Saved Grace's Quarters runtime table:", runtime_paper_path)
    if verbose:
        print(f"[report] Finished Grace's Quarters report in {time.perf_counter() - wall_t0:.1f}s", flush=True)
    return {
        "merged": merged,
        "summary": summary,
        "runtime_components": component_df,
        "runtime": runtime,
        "paper_runtime": paper_runtime,
        "paths": {
            "plot_pdf": plot_pdf,
            "plot_png": plot_png,
            "summary_csv": styled_summary_path,
            "runtime_csv": runtime_paper_path,
            "runtime_latex": runtime_latex_path,
        },
    }
