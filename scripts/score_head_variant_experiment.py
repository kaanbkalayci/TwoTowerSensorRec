from __future__ import annotations

from dataclasses import asdict
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.two_tower_training import (
    TrainConfig,
    build_utility_labels,
    compare_on_common_objectives,
    evaluate_split,
    make_loader,
    make_loss,
    prepare_standardized_data,
    set_all_seeds,
    summarize_results,
)


DEFAULT_PREFIX = "vehicle_sensor_subset_200ms_expanded_features"
DEFAULT_V3_DIR = Path("experiments") / "static_action_two_tower" / "bestmodel_v3"
DEFAULT_OUT_DIR = Path("experiments") / "score_head_variant_two_tower"
DEFAULT_TABLE_DIR = Path("reports") / "tables" / "score_head_variant_two_tower"
REGRET_REL_TOL = 0.03


class TwoTowerScoreVariant(nn.Module):
    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        *,
        hidden: int = 512,
        emb_dim: int = 16,
        depth: int = 2,
        dropout: float = 0.05,
        score_mode: str = "mul_head",
    ) -> None:
        super().__init__()
        self.score_mode = score_mode
        self.context_tower = self._tower(context_dim, hidden, emb_dim, depth, dropout)
        self.action_tower = self._tower(action_dim, hidden, emb_dim, depth, dropout)

        if score_mode == "mul_head":
            self.head = self._head(emb_dim * 3, hidden, dropout)
        elif score_mode == "concat_head":
            self.head = self._head(emb_dim * 2, hidden, dropout)
        elif score_mode == "prod_head":
            self.head = self._head(emb_dim, hidden, dropout)
        elif score_mode == "dot":
            self.head = None
        elif score_mode == "weighted_dot":
            self.head = None
            self.diag_weight = nn.Parameter(torch.ones(emb_dim))
            self.bias = nn.Parameter(torch.zeros(()))
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

    @staticmethod
    def _tower(in_dim: int, hidden: int, emb_dim: int, depth: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(depth - 1, 1)):
            layers.extend(
                [
                    nn.Linear(dim, hidden),
                    nn.ReLU(),
                    nn.LayerNorm(hidden),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden
        layers.append(nn.Linear(dim, emb_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _head(in_dim: int, hidden: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def embed_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.context_tower(context)

    def embed_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_tower(action)

    def score_embeddings(self, c_emb: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        prod = c_emb * a_emb
        if self.score_mode == "dot":
            return prod.sum(dim=-1)
        if self.score_mode == "weighted_dot":
            return (prod * self.diag_weight).sum(dim=-1) + self.bias
        if self.score_mode == "mul_head":
            rep = torch.cat([c_emb, a_emb, prod], dim=-1)
        elif self.score_mode == "concat_head":
            rep = torch.cat([c_emb, a_emb], dim=-1)
        elif self.score_mode == "prod_head":
            rep = prod
        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode}")
        return self.head(rep).squeeze(-1)

    def forward(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.score_embeddings(self.embed_context(context), self.embed_action(action))


def _selector_key(val: dict[str, float], best_regret_so_far: float) -> tuple[float, ...]:
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


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _refined_feature_from_context_name(name: str) -> tuple[int, str] | None:
    match = re.match(r"^n(\d+)_audio_(band_.+_db|band_.+_ratio|low_high_ratio_db|voice_band_ratio|high_band_ratio|band_.+_delta_db)$", name)
    if match:
        return int(match.group(1)), match.group(2)
    return None


def _build_context_from_manifest(
    project_root: Path,
    *,
    prefix: str,
    manifest: dict[str, Any],
) -> tuple[np.ndarray, list[str], dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    processed_dir = project_root / "data" / "processed"
    arrays = np.load(processed_dir / f"{prefix}_arrays.npz", allow_pickle=True)
    meta = _load_json(processed_dir / f"{prefix}_meta.json")
    examples_index = pd.read_csv(processed_dir / f"{prefix}_examples_index.csv")
    examples_index["subset_str"] = examples_index["subset_str"].astype(str)

    C_base = arrays["C_by_time"].astype(np.float32)
    y_examples = arrays["y_examples"].astype(np.float32)
    example_time_id = arrays["example_time_id"].astype(np.int64)
    sequence_times = pd.DatetimeIndex(pd.to_datetime(arrays["sequence_times"]))
    base_names = list(meta["context_feature_names"])
    base_index = {name: i for i, name in enumerate(base_names)}

    refined_path = processed_dir / "vehicle_sensor_subset_200ms_refined_audio_bands_long.csv"
    refined_long = None
    refined_index: dict[tuple[int, str], pd.Series] = {}
    if refined_path.exists():
        refined_long = pd.read_csv(refined_path)
        refined_long["datetime"] = pd.to_datetime(refined_long["datetime"])
        refined_long["node"] = refined_long["node"].astype(int)

    cols = []
    names = []
    for name in manifest["context_feature_names"]:
        if name in base_index:
            cols.append(C_base[:, base_index[name]].astype(np.float32))
            names.append(name)
            continue

        refined_key = _refined_feature_from_context_name(name)
        if refined_key is None or refined_long is None:
            raise KeyError(f"Cannot build context feature {name!r}")
        node, feat = refined_key
        if (node, feat) not in refined_index:
            node_df = refined_long.loc[refined_long["node"].eq(node)].set_index("datetime")
            if feat not in node_df.columns:
                raise KeyError(f"Refined audio cache is missing {feat!r} for node {node}")
            values = node_df[feat].reindex(sequence_times).to_numpy(dtype=float)
            fill = float(np.nanmedian(values[np.isfinite(values)])) if np.isfinite(values).any() else 0.0
            refined_index[(node, feat)] = pd.Series(
                np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill).astype(np.float32),
                index=sequence_times,
            )
        cols.append(refined_index[(node, feat)].to_numpy(dtype=np.float32))
        names.append(name)

    C = np.column_stack(cols).astype(np.float32)
    if C.shape[1] != int(manifest["context_dim"]):
        raise RuntimeError(f"Expected context_dim={manifest['context_dim']}, got {C.shape[1]}")
    return C, names, meta, examples_index, y_examples, example_time_id, arrays["sequence_times"], sequence_times


def load_bestmodel_v3_data(
    project_root: str | Path | None = None,
    *,
    prefix: str = DEFAULT_PREFIX,
    v3_dir: str | Path = DEFAULT_V3_DIR,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    project_root = Path(project_root or Path.cwd())
    if project_root.name == "notebooks":
        project_root = project_root.parent
    v3_dir = project_root / v3_dir
    info = _load_json(v3_dir / "bestmodel_v3_info.json")
    manifest = _load_json(v3_dir / "source_feature_manifest.json")

    C, context_names, meta, examples_index, y_examples, example_time_id, sequence_times_raw, _sequence_index = _build_context_from_manifest(
        project_root,
        prefix=prefix,
        manifest=manifest,
    )

    static_npz = np.load(v3_dir / "static_action_embeddings.npz", allow_pickle=True)
    subset_key = np.asarray(static_npz["subset_key"]).astype(str)
    subset_to_action_id = {key: i for i, key in enumerate(subset_key)}
    action_id = examples_index["subset_str"].map(subset_to_action_id)
    if action_id.isna().any():
        missing = examples_index.loc[action_id.isna(), "subset_str"].unique().tolist()
        raise ValueError(f"Subsets missing from v3 action catalog: {missing[:5]}")
    action_id = action_id.to_numpy(dtype=np.int64)

    action_catalog = static_npz["action_vectors"].astype(np.float32)
    action_names = [str(x) for x in static_npz["static_action_feature_names"]]
    A_examples = action_catalog[action_id].astype(np.float32)

    data_meta = copy.deepcopy(meta)
    data_meta["context_feature_names"] = context_names
    data_meta["context_dim"] = int(C.shape[1])
    data_meta["static_action_feature_names"] = action_names
    data_meta["action_feature_names"] = action_names
    data_meta["action_raw_dim"] = int(A_examples.shape[1])
    data_meta["source_bestmodel_v3_info"] = str(v3_dir / "bestmodel_v3_info.json")
    data_meta["source_feature_manifest"] = str(v3_dir / "source_feature_manifest.json")
    data_meta["score_head_variant_note"] = "Scoring-head architecture sweep using bestmodel_v3 features."

    data = {
        "C_by_time": C,
        "A_examples": A_examples,
        "y_examples": y_examples.copy(),
        "saved_y_examples": y_examples.copy(),
        "example_time_id": example_time_id.copy(),
        "sequence_times": np.asarray(sequence_times_raw, dtype="datetime64[ns]"),
        "examples_index": examples_index.copy(),
        "meta": data_meta,
        "utility_name": "saved",
        "utility_kwargs": {},
    }
    run_manifest = {
        **manifest,
        "context_feature_names": context_names,
        "static_action_feature_names": action_names,
        "source_v3_best_epoch": int(info.get("best_epoch", -1)),
        "source_v3_metrics": info.get("metrics", {}),
        "subset_key": subset_key.tolist(),
    }
    return data, action_catalog, run_manifest


def train_score_variant(
    data: dict[str, Any],
    action_catalog: np.ndarray,
    manifest: dict[str, Any],
    config: TrainConfig,
    *,
    score_mode: str,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    set_all_seeds(config.seed)
    run_dir = output_dir / config.run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_data = dict(data)
    if train_data.get("utility_name") != config.utility_name or train_data.get("utility_kwargs") != (config.utility_kwargs or {}):
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
    model = TwoTowerScoreVariant(
        context_dim=prepared["C_by_time"].shape[1],
        action_dim=prepared["A_examples"].shape[1],
        hidden=config.hidden,
        emb_dim=config.emb_dim,
        depth=config.depth,
        dropout=config.dropout,
        score_mode=score_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = make_loss(config.loss_name)

    best_state = None
    best_key = None
    best_epoch = -1
    best_regret_so_far = float("inf")
    wait = 0
    history_rows = []
    start = time.time()

    print(f"\n=== {config.run_name} ===", flush=True)
    print(
        f"score_mode={score_mode} context_dim={prepared['C_by_time'].shape[1]} "
        f"action_dim={prepared['A_examples'].shape[1]} emb_dim={config.emb_dim}",
        flush=True,
    )

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for c_batch, a_batch, y_batch in train_loader:
            c_batch = c_batch.to(device)
            a_batch = a_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(c_batch, a_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(losses))
        val = evaluate_split(model, prepared, "val", config, device)
        best_regret_so_far = min(best_regret_so_far, float(val["avg_regret"]))
        key = _selector_key(val, best_regret_so_far)
        improved = best_key is None or key < best_key
        if improved:
            best_key = key
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            torch.save(
                {
                    "epoch": int(epoch),
                    "score_mode": score_mode,
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
                "score_mode": score_mode,
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
                f"{time.time() - start:.0f}s",
                flush=True,
            )

        if wait >= config.patience:
            print(f"{config.run_name} early stop at epoch {epoch}; selected epoch {best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No best state selected for {config.run_name}")

    model.load_state_dict(best_state)
    metrics = {
        split: evaluate_split(model, prepared, split, config, device)
        for split in ("train", "val", "test")
    }
    result = {
        "config": config,
        "score_mode": score_mode,
        "model": model,
        "prepared": prepared,
        "history": pd.DataFrame(history_rows),
        "metrics": metrics,
        "best_epoch": int(best_epoch),
        "device": device,
    }

    summary_df = summarize_results([result])
    summary_df.insert(1, "score_mode", score_mode)
    common_eval_df = compare_on_common_objectives(
        [result],
        objectives=[
            {"name": "saved_rational", "utility_name": "saved", "utility_kwargs": {}},
            {"name": "contains_closest", "utility_name": "closest_binary", "utility_kwargs": {}},
            {"name": "rank_discount", "utility_name": "rank_discount", "utility_kwargs": {}},
        ],
        split_names=("train", "val", "test"),
    )
    common_eval_df.insert(1, "score_mode", score_mode)

    a_mu = prepared["standardizers"]["A_mu"]
    a_sigma = prepared["standardizers"]["A_sigma"]
    a_sigma = np.where(np.abs(a_sigma) < 1e-8, 1.0, a_sigma).astype(np.float32)
    action_catalog_std = ((action_catalog - a_mu) / a_sigma).astype(np.float32)
    model.eval()
    with torch.no_grad():
        action_embeddings = model.embed_action(torch.from_numpy(action_catalog_std).to(device)).detach().cpu().numpy().astype(np.float32)

    checkpoint_path = run_dir / "selected_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "score_mode": score_mode,
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
        action_embeddings=action_embeddings,
        action_vectors=action_catalog.astype(np.float32),
        action_vectors_std=action_catalog_std.astype(np.float32),
        subset_key=np.asarray(manifest["subset_key"], dtype=object),
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


def run(
    project_root: str | Path | None = None,
    *,
    max_epochs: int = 150,
    patience: int | None = None,
    log_every: int = 5,
    device: str | None = None,
    score_modes: list[str] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root or Path.cwd())
    if project_root.name == "notebooks":
        project_root = project_root.parent
    out_dir = project_root / DEFAULT_OUT_DIR
    table_dir = project_root / DEFAULT_TABLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    patience = int(max_epochs if patience is None else patience)

    data, action_catalog, manifest = load_bestmodel_v3_data(project_root)
    base_cfg = TrainConfig(
        run_name="placeholder",
        utility_name="saved",
        hidden=512,
        emb_dim=16,
        depth=2,
        dropout=0.05,
        combine_mode="placeholder",
        loss_name="mse",
        lr=5e-4,
        weight_decay=1e-4,
        batch_size=8192,
        max_epochs=int(max_epochs),
        patience=patience,
        seed=22,
        train_frac=0.60,
        val_frac=0.20,
        log_every=int(log_every),
        num_workers=0,
    )
    score_modes = score_modes or ["mul_head", "concat_head", "prod_head", "dot", "weighted_dot"]

    print("PROJECT_ROOT:", project_root, flush=True)
    print("DEVICE:", device, flush=True)
    print("context_dim:", data["C_by_time"].shape[1], "action_dim:", data["A_examples"].shape[1], flush=True)
    print("examples:", len(data["A_examples"]), "times:", len(data["C_by_time"]), flush=True)
    print("score modes:", score_modes, flush=True)

    results = []
    for score_mode in score_modes:
        cfg = copy.deepcopy(base_cfg)
        cfg.combine_mode = score_mode
        cfg.run_name = f"v3_score_{score_mode}_h512_d2_e16"
        result = train_score_variant(
            data,
            action_catalog,
            manifest,
            cfg,
            score_mode=score_mode,
            output_dir=out_dir,
            device=device,
        )
        results.append(result)

    summary = pd.concat([r["summary_df"] for r in results], ignore_index=True)
    common_eval = pd.concat([r["common_eval_df"] for r in results], ignore_index=True)

    summary_path = table_dir / "score_head_variant_summary.csv"
    common_path = table_dir / "score_head_variant_common_eval.csv"
    history_path = table_dir / "score_head_variant_history.csv"
    manifest_path = table_dir / "score_head_variant_feature_manifest.json"
    summary.to_csv(summary_path, index=False)
    common_eval.to_csv(common_path, index=False)
    pd.concat([r["history"].assign(run_name=r["config"].run_name) for r in results], ignore_index=True).to_csv(history_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    comparison_rows = []
    for row in common_eval.itertuples(index=False):
        if row.split == "test":
            comparison_rows.append(row._asdict())
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["eval_objective", "avg_norm_regret", "mean_rank"],
        ascending=[True, True, True],
    )
    comparison_path = table_dir / "score_head_variant_test_objectives.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\nSaved score-head variant tables:")
    print(" ", summary_path)
    print(" ", common_path)
    print(" ", comparison_path)
    return {
        "results": results,
        "summary": summary,
        "common_eval": common_eval,
        "test_objectives": comparison,
        "paths": {
            "out_dir": out_dir,
            "table_dir": table_dir,
            "summary": summary_path,
            "common_eval": common_path,
            "test_objectives": comparison_path,
            "history": history_path,
            "manifest": manifest_path,
        },
    }
