from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from scripts.graces_quarters_rssi_two_tower import (
    DEFAULT_PREFIX,
    _parse_flac_start,
    _static_action_catalog_from_geometry,
)
from scripts.rssi_only_two_tower_baseline import (
    build_per_time_decisions,
    evaluate_indices,
    fit_standardizer,
    predict_scores,
    selector_key,
)
from scripts.two_tower_training import TrainConfig, TwoTowerMLP, chronological_split, set_all_seeds


DEFAULT_MODEL_TAG = "graces_quarters_v3_feature_two_tower"
V3_BANDS = (
    (20, 80),
    (80, 160),
    (160, 400),
    (400, 900),
    (900, 2000),
    (2000, 3500),
    (3500, 6000),
)
V3_BAND_DB_FEATURES = [f"band_{lo}_{hi}_db" for lo, hi in V3_BANDS]
EPS = np.finfo(np.float32).eps


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
    start_time = _parse_flac_start(flac_path, node)
    info = sf.info(flac_path)
    frame_len = int(round(info.samplerate * sample_ms / 1000.0))
    if frame_len <= 0:
        raise ValueError("frame_len must be positive")

    window = np.hanning(frame_len).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / info.samplerate).astype(np.float32)
    rows: list[pd.DataFrame] = []
    blocksize = frame_len * frames_per_block
    frame_offset = 0

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
        for lo, hi in V3_BANDS:
            band = _band_power(power, freqs, lo, hi)
            data[f"band_{lo}_{hi}_db"] = (10.0 * np.log10(band + EPS)).astype(np.float32)

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


def build_v3_refined_audio_cache(
    project_root: str | Path | None = None,
    *,
    prefix: str = DEFAULT_PREFIX,
    sample_ms: int = 200,
    force: bool = False,
) -> pd.DataFrame:
    project_root = Path(project_root or Path.cwd())
    if project_root.name == "notebooks":
        project_root = project_root.parent
    processed_dir = project_root / "data" / "processed"
    raw_dir = project_root / "data" / "raw" / "Graces_Quarters"
    cache_path = processed_dir / f"{prefix}_v3_refined_audio_bands_long.csv"

    with open(processed_dir / f"{prefix}_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    ordered_nodes = [int(x) for x in meta["ordered_nodes"]]

    required = {"datetime", "node", *V3_BAND_DB_FEATURES}
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path)
        if required.issubset(cached.columns):
            cached["datetime"] = pd.to_datetime(cached["datetime"])
            cached["node"] = cached["node"].astype(int)
            print("Loaded Graces v3 refined-band cache:", cache_path, cached.shape)
            return cached

    pieces = []
    for node in ordered_nodes:
        matches = sorted(raw_dir.glob(f"*_{node}_respeaker.flac"))
        if not matches:
            raise FileNotFoundError(f"Missing FLAC for node {node} in {raw_dir}")
        node_df = _refined_band_db_frames(matches[0], node=node, sample_ms=sample_ms)
        pieces.append(node_df)
        print(f"refined bands: node {node}, rows={len(node_df)}, file={matches[0].name}")

    refined = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["datetime", "node"])
        .drop_duplicates(["datetime", "node"], keep="first")
        .reset_index(drop=True)
    )
    refined.to_csv(cache_path, index=False)
    print("Saved Graces v3 refined-band cache:", cache_path, refined.shape)
    return refined


def _make_v3_context_matrix(
    *,
    c_full: np.ndarray,
    base_context_names: list[str],
    sequence_times: pd.DatetimeIndex,
    ordered_nodes: list[int],
    refined_long: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    base_context_index = {name: i for i, name in enumerate(base_context_names)}
    pieces: list[np.ndarray] = []
    names: list[str] = []

    for node in ordered_nodes:
        for local in ("sensor_x_norm", "sensor_y_norm"):
            name = f"n{node}_{local}"
            if name not in base_context_index:
                raise KeyError(f"Missing base context feature {name}")
            pieces.append(c_full[:, base_context_index[name]].astype(np.float32))
            names.append(name)

    refined = refined_long.copy()
    refined["datetime"] = pd.to_datetime(refined["datetime"])
    refined["node"] = refined["node"].astype(int)
    sequence_index = pd.DatetimeIndex(sequence_times)

    for node in ordered_nodes:
        node_df = refined.loc[refined["node"].eq(node)].set_index("datetime")
        if node_df.empty:
            raise KeyError(f"Missing refined audio rows for node {node}")
        for feat in V3_BAND_DB_FEATURES:
            values = node_df[feat].reindex(sequence_index).to_numpy(dtype=float)
            finite = np.isfinite(values)
            fill = float(np.nanmedian(values[finite])) if finite.any() else 0.0
            pieces.append(np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill).astype(np.float32))
            names.append(f"n{node}_audio_{feat}")

    return np.column_stack(pieces).astype(np.float32), names


def train_v3_feature_two_tower(
    project_root: str | Path | None = None,
    *,
    prefix: str = DEFAULT_PREFIX,
    model_tag: str = DEFAULT_MODEL_TAG,
    sample_ms: int = 200,
    force_refined_cache: bool = False,
    max_epochs: int = 250,
    patience: int | None = None,
    log_every: int = 5,
    seed: int = 22,
    lr: float = 5e-4,
    dropout: float = 0.05,
    weight_decay: float = 1e-4,
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
    sequence_times = pd.to_datetime(arrays["sequence_times"])
    ordered_nodes = [int(x) for x in meta["ordered_nodes"]]
    base_context_names = list(meta["context_feature_names"])

    refined = build_v3_refined_audio_cache(
        project_root,
        prefix=prefix,
        sample_ms=sample_ms,
        force=force_refined_cache,
    )
    c_v3, context_names = _make_v3_context_matrix(
        c_full=c_full,
        base_context_names=base_context_names,
        sequence_times=sequence_times,
        ordered_nodes=ordered_nodes,
        refined_long=refined,
    )

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
        dropout=dropout,
        combine_mode="mul_only",
        loss_name="mse",
        lr=lr,
        weight_decay=weight_decay,
        batch_size=8192,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        train_frac=0.60,
        val_frac=0.20,
        log_every=log_every,
        num_workers=0,
    )

    split = chronological_split(
        example_time_id,
        n_times=len(c_v3),
        train_frac=config.train_frac,
        val_frac=config.val_frac,
    )
    c_mu, c_sigma = fit_standardizer(c_v3[split["train_time_ids"]])
    a_mu, a_sigma = fit_standardizer(a_examples[split["train"]])
    c_std = ((c_v3 - c_mu) / c_sigma).astype(np.float32)
    a_std = ((a_examples - a_mu) / a_sigma).astype(np.float32)
    a_catalog_std = ((a_catalog - a_mu) / a_sigma).astype(np.float32)

    print("Grace's Quarters v3-feature two-tower")
    print("  context_dim:", c_v3.shape[1], "=", len(ordered_nodes), "nodes x (2 coords + 7 band dB)")
    print("  action_dim:", a_examples.shape[1], "=", len(ordered_nodes), "masks + 3 coordinate slots + subset_size")
    print("  candidate subsets:", len(subset_key))
    print("  examples:", len(y_examples))

    set_all_seeds(config.seed)
    rng = np.random.default_rng(config.seed)
    model = TwoTowerMLP(
        context_dim=c_v3.shape[1],
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

        val_scores = predict_scores(
            model,
            c_std,
            a_std,
            example_time_id,
            split["val"],
            config.batch_size * 2,
            device,
        )
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
    all_scores = np.empty_like(y_examples, dtype=np.float32)
    split_by_time: dict[int, str] = {}
    for split_name in ("train", "val", "test"):
        for time_id in split[f"{split_name}_time_ids"]:
            split_by_time[int(time_id)] = split_name
        scores = predict_scores(
            model,
            c_std,
            a_std,
            example_time_id,
            split[split_name],
            config.batch_size * 2,
            device,
        )
        all_scores[split[split_name]] = scores
        metrics[split_name] = evaluate_indices(split[split_name], scores, y_examples, example_time_id, contains_examples)
        split_rows.append(
            {
                "split": split_name,
                "n_examples": int(len(split[split_name])),
                "best_epoch": int(best_epoch),
                **metrics[split_name],
            }
        )

    with torch.no_grad():
        action_embeddings = (
            model.embed_action(torch.from_numpy(a_catalog_std.astype(np.float32)).to(device))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    history_df = pd.DataFrame(history)
    metrics_df = pd.DataFrame(split_rows)
    decisions = build_per_time_decisions(examples_index, y_examples, all_scores, split_by_time)

    history_df.to_csv(table_dir / f"{model_tag}_history.csv", index=False)
    metrics_df.to_csv(table_dir / f"{model_tag}_metrics.csv", index=False)
    decisions.to_csv(table_dir / f"{model_tag}_decisions.csv", index=False)
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
            "context_dim": int(c_v3.shape[1]),
            "action_dim": int(a_examples.shape[1]),
            "best_epoch": int(best_epoch),
            "metrics": metrics,
            "standardizers": {"C_mu": c_mu, "C_sigma": c_sigma, "A_mu": a_mu, "A_sigma": a_sigma},
            "context_feature_names": context_names,
            "static_action_feature_names": action_names,
            "prefix": prefix,
            "ordered_nodes": ordered_nodes,
            "feature_recipe": "bestmodel_v3_generalized_to_graces_nodes",
            "refined_audio_features_added": V3_BAND_DB_FEATURES,
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
                "refined_audio_features_added": V3_BAND_DB_FEATURES,
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
        "decisions": decisions,
        "paths": {"out_dir": out_dir, "table_dir": table_dir},
        "context_feature_names": context_names,
        "static_action_feature_names": action_names,
    }
