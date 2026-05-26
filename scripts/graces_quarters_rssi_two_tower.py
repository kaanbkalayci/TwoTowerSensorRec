from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import soundfile as sf
import torch

from scripts.processed_feature_builder import build_processed_two_tower_data, save_processed_two_tower_data
from scripts.rssi_only_two_tower_baseline import (
    EPS,
    evaluate_indices,
    fit_standardizer,
    predict_scores,
    selector_key,
)
from scripts.two_tower_training import TrainConfig, TwoTowerMLP, chronological_split, set_all_seeds


DEFAULT_PREFIX = "graces_quarters_rssi_subset_200ms"
DEFAULT_MODEL_TAG = "graces_quarters_rssi_only_two_tower"


def _pick_utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _parse_flac_start(path: Path, node: int) -> datetime:
    match = re.match(rf"(\d{{8}})_(\d{{6}})_.*_{node}_respeaker\.flac$", path.name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse FLAC start time from {path.name}")
    return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y%m%d %H%M%S")


def _read_vehicle_gps(gps_path: Path, sample_ms: int) -> pd.DataFrame:
    gps = pd.read_csv(gps_path, header=None)
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(gps.iloc[:, 0], errors="coerce"),
            "Latitude": pd.to_numeric(gps.iloc[:, 1], errors="coerce"),
            "Longitude": pd.to_numeric(gps.iloc[:, 2], errors="coerce"),
        }
    )
    out = out.dropna(subset=["datetime", "Latitude", "Longitude"])
    out = out.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    resampled = out.resample(f"{sample_ms}ms").mean().interpolate(method="time")
    return resampled.dropna(subset=["Latitude", "Longitude"])


def _node_audio_power_db(path: Path, node: int, sample_ms: int, frames_per_block: int = 512) -> pd.DataFrame:
    start_time = _parse_flac_start(path, node)
    info = sf.info(path)
    frame_len = int(round(info.samplerate * sample_ms / 1000.0))
    if frame_len <= 0:
        raise ValueError("frame_len must be positive")

    power_chunks: list[np.ndarray] = []
    blocksize = frame_len * frames_per_block
    for block in sf.blocks(path, blocksize=blocksize, dtype="float32", always_2d=True):
        hi = min(block.shape[1], 5)
        lo = 1 if block.shape[1] > 1 else 0
        mono = block[:, lo:hi].mean(axis=1).astype(np.float32, copy=False)
        n_frames = len(mono) // frame_len
        if n_frames <= 0:
            continue
        frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
        power_chunks.append(np.mean(frames * frames, axis=1).astype(np.float32))

    if not power_chunks:
        raise RuntimeError(f"No audio frames extracted from {path}")

    power = np.concatenate(power_chunks)
    rssi_db = 10.0 * np.log10(np.maximum(power, np.finfo(np.float32).eps))
    index = pd.date_range(start=start_time, periods=len(rssi_db), freq=f"{sample_ms}ms")
    return pd.DataFrame({f"rpi{node}": rssi_db.astype(np.float32)}, index=index)


def _load_rssi_matrix(raw_dir: Path, nodes: list[int], sample_ms: int) -> pd.DataFrame:
    pieces = []
    for node in nodes:
        matches = sorted(raw_dir.glob(f"*_{node}_respeaker.flac"))
        if not matches:
            raise FileNotFoundError(f"Missing FLAC for node {node} in {raw_dir}")
        df = _node_audio_power_db(matches[0], node=node, sample_ms=sample_ms)
        pieces.append(df)
        print(f"node {node}: {len(df)} frames, {df.index.min()} to {df.index.max()}")
    combined = pd.concat(pieces, axis=1).sort_index()
    return combined[~combined.index.duplicated(keep="first")]


def _valid_indices_from_motion(gdf_cleaned: gpd.GeoDataFrame, sample_ms: int) -> list[int]:
    valid = []
    threshold_m = 0.125 * sample_ms / 1000.0 * 5.0 / 3.0
    for i in range(1, len(gdf_cleaned)):
        if (gdf_cleaned.index[i] - gdf_cleaned.index[i - 1]) != pd.Timedelta(milliseconds=sample_ms):
            continue
        if gdf_cleaned.iloc[i].geometry.distance(gdf_cleaned.iloc[i - 1].geometry) >= threshold_m:
            valid.append(i)
    return valid


def load_graces_quarters(
    project_root: str | Path | None = None,
    *,
    dataset_name: str = "Graces_Quarters",
    sample_ms: int = 200,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, list[int], gpd.GeoDataFrame]:
    project_root = Path(project_root or Path.cwd())
    if project_root.name == "notebooks":
        project_root = project_root.parent
    raw_dir = project_root / "data" / "raw" / dataset_name
    nodes_path = raw_dir / "nodes_gps.csv"
    gps_path = raw_dir / "ATV2-20250812-105042.csv"

    nodes_df = pd.read_csv(nodes_path)
    nodes = [int(x) for x in nodes_df["Node #"].tolist()]
    rssi_df = _load_rssi_matrix(raw_dir, nodes, sample_ms=sample_ms)
    gps_df = _read_vehicle_gps(gps_path, sample_ms=sample_ms)

    merged = rssi_df.merge(gps_df, left_index=True, right_index=True, how="inner")
    merged = merged.dropna(subset=[f"rpi{node}" for node in nodes] + ["Latitude", "Longitude"])

    mean_lat = float(nodes_df["Lat"].mean())
    mean_lon = float(nodes_df["Lon"].mean())
    local_epsg = _pick_utm_epsg(mean_lat, mean_lon)

    gdf_nodes = gpd.GeoDataFrame(
        nodes_df.copy(),
        geometry=[Point(lon, lat) for lat, lon in zip(nodes_df["Lat"], nodes_df["Lon"])],
        crs="EPSG:4326",
    ).to_crs(epsg=local_epsg)

    merged_reset = merged.reset_index(names="datetime")
    gdf_vehicle = gpd.GeoDataFrame(
        merged_reset,
        geometry=gpd.points_from_xy(merged_reset["Longitude"], merged_reset["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=local_epsg)
    gdf_cleaned = gdf_vehicle.drop(columns=["Latitude", "Longitude"]).set_index("datetime").sort_index()

    for _, row in gdf_nodes.iterrows():
        node = int(row["Node #"])
        gdf_cleaned[f"distance_to_{node}"] = gdf_cleaned.geometry.distance(row.geometry)

    normalized_cleaned = gdf_cleaned.drop(columns="geometry").copy()
    valid_indices = _valid_indices_from_motion(gdf_cleaned, sample_ms=sample_ms)

    print("\nGraces_Quarters RSSI instance")
    print("nodes:", nodes)
    print("rows:", len(gdf_cleaned), "valid:", len(valid_indices))
    print("motion threshold m/step:", 0.125 * sample_ms / 1000.0 * 5.0 / 3.0)
    print("time range:", gdf_cleaned.index.min(), "to", gdf_cleaned.index.max())
    print("projected CRS:", f"EPSG:{local_epsg}")
    return gdf_cleaned, normalized_cleaned, valid_indices, gdf_nodes


def build_and_save_processed(
    project_root: str | Path | None = None,
    *,
    prefix: str = DEFAULT_PREFIX,
    sample_ms: int = 200,
    max_subset_size: int = 3,
    force: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root or Path.cwd())
    if project_root.name == "notebooks":
        project_root = project_root.parent
    processed_dir = project_root / "data" / "processed"
    arrays_path = processed_dir / f"{prefix}_arrays.npz"
    meta_path = processed_dir / f"{prefix}_meta.json"
    if arrays_path.exists() and meta_path.exists() and not force:
        print("Using existing processed artifacts:", arrays_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        arrays = np.load(arrays_path, allow_pickle=True)
        return {"paths": {"arrays_npz": arrays_path, "meta_json": meta_path}, "meta": meta, "arrays": arrays}

    gdf_cleaned, normalized_cleaned, valid_indices, gdf_nodes = load_graces_quarters(project_root, sample_ms=sample_ms)
    node_list = [int(x) for x in gdf_nodes["Node #"].tolist()]
    processed = build_processed_two_tower_data(
        gdf_cleaned,
        gdf_nodes,
        valid_indices,
        node_list,
        audio_feature_long=None,
        history_steps=5,
        max_subset_size=max_subset_size,
        context_audio_features=[],
        action_audio_features=[],
        include_audio_derived_features=False,
        verbose=True,
        progress_every=1000,
    )
    paths = save_processed_two_tower_data(processed, processed_dir, prefix=prefix)
    print("Saved processed Graces_Quarters artifacts:")
    for key, value in paths.items():
        print(f"  {key}: {value}")
    return {
        "gdf_cleaned": gdf_cleaned,
        "normalized_cleaned": normalized_cleaned,
        "valid_indices": valid_indices,
        "gdf_nodes": gdf_nodes,
        "processed": processed,
        "paths": paths,
        "meta": processed["meta"],
    }


def _static_action_catalog_from_geometry(
    examples_index: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    ordered_nodes: list[int],
    *,
    max_subset_size: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    node_to_row = {
        int(row["node"]): row
        for _, row in sensor_geometry.iterrows()
    }
    missing = [node for node in ordered_nodes if node not in node_to_row]
    if missing:
        raise KeyError(f"Sensor geometry missing nodes: {missing}")

    subset_key = sorted(examples_index["subset_str"].astype(str).unique(), key=lambda s: (s.count("-"), [int(x) for x in s.split("-")]))
    subset_to_id = {subset: i for i, subset in enumerate(subset_key)}
    example_action_id = examples_index["subset_str"].astype(str).map(subset_to_id).to_numpy(dtype=np.int64)

    action_names = (
        [f"mask_n{node}" for node in ordered_nodes]
        + [
            f"slot{slot}_sensor_{axis}_norm"
            for slot in range(1, max_subset_size + 1)
            for axis in ("x", "y")
        ]
        + ["subset_size"]
    )

    catalog_rows = []
    node_to_pos = {node: i for i, node in enumerate(ordered_nodes)}
    for subset in subset_key:
        subset_nodes = [int(x) for x in subset.split("-")]
        subset_nodes = sorted(subset_nodes)
        mask = np.zeros(len(ordered_nodes), dtype=np.float32)
        for node in subset_nodes:
            mask[node_to_pos[node]] = 1.0

        slot_values: list[float] = []
        for slot_idx in range(max_subset_size):
            if slot_idx < len(subset_nodes):
                row = node_to_row[subset_nodes[slot_idx]]
                slot_values.extend([float(row["sensor_x_norm"]), float(row["sensor_y_norm"])])
            else:
                slot_values.extend([0.0, 0.0])

        catalog_rows.append(
            np.asarray(
                mask.tolist()
                + slot_values
                + [float(len(subset_nodes))],
                dtype=np.float32,
            )
        )

    catalog = np.stack(catalog_rows).astype(np.float32)
    return catalog[example_action_id], catalog, np.asarray(subset_key, dtype=object), action_names


def train_rssi_only_two_tower(
    project_root: str | Path | None = None,
    *,
    prefix: str = DEFAULT_PREFIX,
    model_tag: str = DEFAULT_MODEL_TAG,
    max_epochs: int = 250,
    patience: int | None = None,
    log_every: int = 5,
    seed: int = 22,
    device: str | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root or Path.cwd())
    if project_root.name == "notebooks":
        project_root = project_root.parent
    processed_dir = project_root / "data" / "processed"
    out_dir = project_root / "experiments" / model_tag
    table_dir = project_root / "reports" / "tables" / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    patience = max_epochs if patience is None else patience

    arrays = np.load(processed_dir / f"{prefix}_arrays.npz", allow_pickle=True)
    with open(processed_dir / f"{prefix}_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    examples_index = pd.read_csv(processed_dir / f"{prefix}_examples_index.csv")
    examples_index["subset_str"] = examples_index["subset_str"].astype(str)
    examples_index["datetime"] = pd.to_datetime(examples_index["datetime"])

    c_full = arrays["C_by_time"].astype(np.float32)
    y_examples = arrays["y_examples"].astype(np.float32)
    example_time_id = arrays["example_time_id"].astype(np.int64)
    contains_examples = arrays["example_contains_closest"].astype(np.float32)

    ordered_nodes = [int(x) for x in meta["ordered_nodes"]]
    context_names_full = list(meta["context_feature_names"])
    context_names = [name for node in ordered_nodes for name in (f"n{node}_sensor_x_norm", f"n{node}_sensor_y_norm", f"n{node}_rssi_db")]

    c_idx = [context_names_full.index(name) for name in context_names]
    c_rssi = c_full[:, c_idx].astype(np.float32)
    sensor_geometry = pd.read_csv(processed_dir / f"{prefix}_sensor_geometry.csv")
    a_examples, a_catalog, subset_key, action_names = _static_action_catalog_from_geometry(
        examples_index,
        sensor_geometry,
        ordered_nodes,
        max_subset_size=int(meta.get("max_subset_size", 3)),
    )

    config = TrainConfig(
        run_name=f"{model_tag}_h512_d2_e16",
        utility_name="saved",
        hidden=512,
        emb_dim=16,
        depth=2,
        dropout=0.05,
        combine_mode="mul_only",
        loss_name="mse",
        lr=5e-4,
        weight_decay=1e-4,
        batch_size=8192,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        train_frac=0.60,
        val_frac=0.20,
        log_every=log_every,
        num_workers=0,
    )

    split = chronological_split(example_time_id, n_times=len(c_rssi), train_frac=config.train_frac, val_frac=config.val_frac)
    c_mu, c_sigma = fit_standardizer(c_rssi[split["train_time_ids"]])
    a_mu, a_sigma = fit_standardizer(a_examples[split["train"]])
    c_std = ((c_rssi - c_mu) / c_sigma).astype(np.float32)
    a_std = ((a_examples - a_mu) / a_sigma).astype(np.float32)
    a_catalog_std = ((a_catalog - a_mu) / a_sigma).astype(np.float32)

    set_all_seeds(config.seed)
    rng = np.random.default_rng(config.seed)
    model = TwoTowerMLP(
        context_dim=c_rssi.shape[1],
        action_dim=a_examples.shape[1],
        hidden=config.hidden,
        emb_dim=config.emb_dim,
        depth=config.depth,
        dropout=config.dropout,
        combine_mode=config.combine_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_state = None
    best_epoch = -1
    best_key = None
    best_regret_so_far = np.inf
    wait = 0
    history = []
    start_clock = time.time()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        perm = rng.permutation(split["train"])
        for start in range(0, len(perm), config.batch_size):
            idx = perm[start : start + config.batch_size]
            c = torch.from_numpy(c_std[example_time_id[idx]].astype(np.float32)).to(device)
            a = torch.from_numpy(a_std[idx].astype(np.float32)).to(device)
            y = torch.from_numpy(y_examples[idx].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(c, a)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_scores = predict_scores(model, c_std, a_std, example_time_id, split["val"], config.batch_size * 2, device)
        val = evaluate_indices(split["val"], val_scores, y_examples, example_time_id, contains_examples)
        best_regret_so_far = min(best_regret_so_far, float(val["avg_regret"]))
        key = selector_key(val, best_regret_so_far)
        improved = best_key is None or key < best_key
        if improved:
            best_key = key
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        if epoch == 1 or epoch % config.log_every == 0 or improved or epoch == config.max_epochs:
            star = "*" if improved else " "
            elapsed = time.time() - start_clock
            print(
                f"{config.run_name} ep {epoch:03d}/{config.max_epochs:03d}{star} "
                f"loss={row['train_loss']:.5f} val_rmse={val['rmse']:.4f} "
                f"top1={val['top1']:.3f} top3={val['top3']:.3f} "
                f"contains={val['contains_closest']:.3f} rank={val['mean_rank']:.2f} "
                f"reg={val['avg_regret']:.4f} {elapsed:.0f}s"
            )
        if wait >= config.patience:
            print(f"early stop at epoch {epoch}; selected epoch {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best state selected.")
    model.load_state_dict(best_state)

    metrics = {}
    split_rows = []
    for split_name in ("train", "val", "test"):
        scores = predict_scores(model, c_std, a_std, example_time_id, split[split_name], config.batch_size * 2, device)
        metrics[split_name] = evaluate_indices(split[split_name], scores, y_examples, example_time_id, contains_examples)
        split_rows.append({"split": split_name, "n_examples": int(len(split[split_name])), "best_epoch": int(best_epoch), **metrics[split_name]})

    with torch.no_grad():
        action_embeddings = model.embed_action(torch.from_numpy(a_catalog_std.astype(np.float32)).to(device)).detach().cpu().numpy().astype(np.float32)

    history_df = pd.DataFrame(history)
    metrics_df = pd.DataFrame(split_rows)
    history_df.to_csv(table_dir / f"{model_tag}_history.csv", index=False)
    metrics_df.to_csv(table_dir / f"{model_tag}_metrics.csv", index=False)
    np.savez(
        out_dir / "static_action_embeddings.npz",
        action_embeddings=action_embeddings,
        action_vectors=a_catalog,
        action_vectors_std=a_catalog_std,
        subset_key=subset_key,
        static_action_feature_names=np.asarray(action_names, dtype=object),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "context_dim": int(c_rssi.shape[1]),
            "action_dim": int(a_examples.shape[1]),
            "best_epoch": int(best_epoch),
            "metrics": metrics,
            "standardizers": {"C_mu": c_mu, "C_sigma": c_sigma, "A_mu": a_mu, "A_sigma": a_sigma},
            "context_feature_names": context_names,
            "static_action_feature_names": action_names,
            "prefix": prefix,
            "ordered_nodes": ordered_nodes,
        },
        out_dir / f"{model_tag}_checkpoint.pt",
    )
    with open(out_dir / f"{model_tag}_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_tag": model_tag,
                "prefix": prefix,
                "best_epoch": int(best_epoch),
                "context_feature_names": context_names,
                "static_action_feature_names": action_names,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    print("Saved model artifacts:", out_dir)
    print("Saved tables:", table_dir)
    return {
        "model": model,
        "history": history_df,
        "metrics": metrics_df,
        "paths": {"out_dir": out_dir, "table_dir": table_dir},
        "context_feature_names": context_names,
        "static_action_feature_names": action_names,
    }
