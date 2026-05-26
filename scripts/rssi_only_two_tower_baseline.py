from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts.two_tower_training import TrainConfig, TwoTowerMLP, chronological_split, set_all_seeds


PREFIX = "vehicle_sensor_subset_200ms_expanded_features"
MODEL_TAG = "rssi_only_two_tower"
REGRET_REL_TOL = 0.03
EPS = 1e-12


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0).astype(np.float32)
    sigma = x.std(axis=0).astype(np.float32)
    sigma[np.abs(sigma) < 1e-8] = 1.0
    return mu, sigma


@torch.no_grad()
def predict_scores(
    model: TwoTowerMLP,
    c_std: np.ndarray,
    a_std: np.ndarray,
    y_time_id: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    scores = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        idx = indices[start:end]
        c = torch.from_numpy(c_std[y_time_id[idx]].astype(np.float32)).to(device)
        a = torch.from_numpy(a_std[idx].astype(np.float32)).to(device)
        scores[start:end] = model(c, a).detach().cpu().numpy().astype(np.float32)
    return scores


def evaluate_indices(
    indices: np.ndarray,
    scores: np.ndarray,
    y_examples: np.ndarray,
    example_time_id: np.ndarray,
    contains_examples: np.ndarray,
) -> dict[str, float]:
    y = y_examples[indices].astype(float)
    err = scores.astype(float) - y
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= EPS else float(1.0 - ss_res / ss_tot)

    df = pd.DataFrame(
        {
            "time_id": example_time_id[indices].astype(int),
            "y": y,
            "score": scores.astype(float),
            "contains": contains_examples[indices].astype(float),
        }
    )
    top1_hits = []
    top3_hits = []
    ranks = []
    regrets = []
    norm_regrets = []
    contains_hits = []
    for _, group in df.groupby("time_id", sort=False):
        yt = group["y"].to_numpy(dtype=float)
        sc = group["score"].to_numpy(dtype=float)
        pred_pos = int(np.argmax(sc))
        selected_value = float(yt[pred_pos])
        best_value = float(np.max(yt))
        worst_value = float(np.min(yt))
        rank = int(np.sum(yt > selected_value + 1e-10)) + 1
        regret = best_value - selected_value
        top1_hits.append(float(rank == 1))
        top3_hits.append(float(rank <= 3))
        ranks.append(rank)
        regrets.append(regret)
        norm_regrets.append(0.0 if best_value - worst_value <= EPS else regret / (best_value - worst_value))
        contains_hits.append(float(group["contains"].to_numpy(dtype=float)[pred_pos] > 0.5))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "top1": float(np.mean(top1_hits)),
        "top3": float(np.mean(top3_hits)),
        "mean_rank": float(np.mean(ranks)),
        "avg_regret": float(np.mean(regrets)),
        "avg_norm_regret": float(np.mean(norm_regrets)),
        "contains_closest": float(np.mean(contains_hits)),
    }


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


def build_per_time_decisions(
    examples_index: pd.DataFrame,
    y_examples: np.ndarray,
    scores: np.ndarray,
    split_by_time: dict[int, str],
) -> pd.DataFrame:
    score_df = examples_index.copy()
    score_df["score"] = scores.astype(np.float32)
    score_df["saved_utility"] = y_examples.astype(np.float32)
    rows = []
    for tid, group in score_df.groupby("time_id", sort=False):
        y = group["saved_utility"].to_numpy(dtype=float)
        sc = group["score"].to_numpy(dtype=float)
        pred_pos = int(np.argmax(sc))
        selected = group.iloc[pred_pos]
        selected_value = float(y[pred_pos])
        sorted_y = np.sort(y)[::-1]
        best_value = float(sorted_y[0])
        second_value = float(sorted_y[1]) if len(sorted_y) > 1 else best_value
        worst_value = float(sorted_y[-1])
        rank = int(np.sum(y > selected_value + 1e-10)) + 1
        regret = best_value - selected_value
        rows.append(
            {
                "time_id": int(tid),
                "datetime": selected["datetime"],
                "split": split_by_time[int(tid)],
                "selected_subset": str(selected["subset_str"]),
                "selected_subset_size": int(selected["subset_size"]),
                "closest_node": int(selected["closest_node"]),
                "signal_top1_node": int(selected["signal_top1_node"]) if "signal_top1_node" in selected.index else np.nan,
                "contains_closest": float(selected["contains_closest_node"] > 0.5),
                "saved_top1": float(rank == 1),
                "saved_top3": float(rank <= 3),
                "saved_rank": float(rank),
                "selected_value": selected_value,
                "best_value": best_value,
                "second_best_value": second_value,
                "worst_value": worst_value,
                "regret": regret,
                "norm_regret": regret / max(best_value - worst_value, EPS),
                "true_top1_top2_gap": best_value - second_value,
                "selected_score": float(sc[pred_pos]),
            }
        )
    out = pd.DataFrame(rows).sort_values("time_id").reset_index(drop=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out


def run(
    project_root: str | Path | None = None,
    device: str | None = None,
    max_epochs: int = 250,
    patience: int | None = None,
    log_every: int = 5,
    seed: int = 22,
) -> dict[str, Any]:
    project_root = Path(project_root or Path.cwd())
    if project_root.name == "notebooks":
        project_root = project_root.parent

    processed_dir = project_root / "data" / "processed"
    best_v3_dir = project_root / "experiments" / "static_action_two_tower" / "bestmodel_v3"
    out_dir = project_root / "experiments" / "static_action_two_tower" / MODEL_TAG
    table_dir = project_root / "reports" / "tables" / "bestmodel_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    patience = max_epochs if patience is None else patience

    arrays = np.load(processed_dir / f"{PREFIX}_arrays.npz", allow_pickle=True)
    with open(processed_dir / f"{PREFIX}_meta.json", "r", encoding="utf-8") as f:
        base_meta = json.load(f)
    with open(best_v3_dir / "source_feature_manifest.json", "r", encoding="utf-8") as f:
        v3_manifest = json.load(f)

    try:
        v3_checkpoint = torch.load(best_v3_dir / "bestmodel_v3_complete_checkpoint.pt", map_location="cpu", weights_only=False)
    except TypeError:
        v3_checkpoint = torch.load(best_v3_dir / "bestmodel_v3_complete_checkpoint.pt", map_location="cpu")
    v3_config = TrainConfig(**v3_checkpoint["config"])

    examples_index = pd.read_csv(processed_dir / f"{PREFIX}_examples_index.csv")
    examples_index["subset_str"] = examples_index["subset_str"].astype(str)
    examples_index["datetime"] = pd.to_datetime(examples_index["datetime"])

    c_base = arrays["C_by_time"].astype(np.float32)
    y_examples = arrays["y_examples"].astype(np.float32)
    example_time_id = arrays["example_time_id"].astype(np.int64)
    ordered_nodes = [int(node) for node in base_meta["ordered_nodes"]]
    base_context_names = list(base_meta["context_feature_names"])
    base_context_index = {name: i for i, name in enumerate(base_context_names)}

    context_names = []
    for node in ordered_nodes:
        context_names.extend([f"n{node}_sensor_x_norm", f"n{node}_sensor_y_norm", f"n{node}_rssi_db"])
    missing_context = [name for name in context_names if name not in base_context_index]
    if missing_context:
        raise KeyError(f"Missing RSSI-only context columns: {missing_context}")
    c_rssi = np.column_stack([c_base[:, base_context_index[name]] for name in context_names]).astype(np.float32)

    static_npz = np.load(best_v3_dir / "static_action_embeddings.npz", allow_pickle=True)
    subset_key = np.asarray(static_npz["subset_key"]).astype(str)
    subset_to_action_id = {key: i for i, key in enumerate(subset_key)}
    action_id_all = examples_index["subset_str"].map(subset_to_action_id).to_numpy()
    if pd.isna(action_id_all).any():
        missing = examples_index.loc[pd.isna(action_id_all), "subset_str"].unique().tolist()
        raise ValueError(f"Subsets missing from static action catalog: {missing}")
    action_id_all = action_id_all.astype(np.int64)
    action_names = list(v3_manifest["static_action_feature_names"])
    a_catalog = static_npz["action_vectors"].astype(np.float32)
    a_examples = a_catalog[action_id_all].astype(np.float32)

    config = TrainConfig(
        run_name="rssi_only_context_h512_d2_e16",
        utility_name="saved",
        hidden=v3_config.hidden,
        emb_dim=v3_config.emb_dim,
        depth=v3_config.depth,
        dropout=v3_config.dropout,
        combine_mode=v3_config.combine_mode,
        loss_name="mse",
        lr=v3_config.lr,
        weight_decay=v3_config.weight_decay,
        batch_size=v3_config.batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        train_frac=v3_config.train_frac,
        val_frac=v3_config.val_frac,
        log_every=log_every,
        num_workers=0,
    )

    split = chronological_split(
        example_time_id,
        n_times=len(c_rssi),
        train_frac=config.train_frac,
        val_frac=config.val_frac,
    )
    c_mu, c_sigma = fit_standardizer(c_rssi[split["train_time_ids"]])
    a_mu, a_sigma = fit_standardizer(a_examples[split["train"]])
    c_std = ((c_rssi - c_mu) / c_sigma).astype(np.float32)
    a_std = ((a_examples - a_mu) / a_sigma).astype(np.float32)
    a_catalog_std = ((a_catalog - a_mu) / a_sigma).astype(np.float32)
    contains_examples = examples_index["contains_closest_node"].to_numpy(dtype=np.float32)

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
    start_time = time.time()

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
            elapsed = time.time() - start_time
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
        raise RuntimeError("No best state selected for RSSI-only two-tower baseline.")
    model.load_state_dict(best_state)

    metrics = {}
    for split_name in ("train", "val", "test"):
        scores = predict_scores(model, c_std, a_std, example_time_id, split[split_name], config.batch_size * 2, device)
        metrics[split_name] = evaluate_indices(split[split_name], scores, y_examples, example_time_id, contains_examples)

    all_indices = np.arange(len(y_examples), dtype=np.int64)
    all_scores = predict_scores(model, c_std, a_std, example_time_id, all_indices, config.batch_size * 2, device)
    split_by_time = {}
    for split_name in ("train", "val", "test"):
        for tid in split[f"{split_name}_time_ids"]:
            split_by_time[int(tid)] = split_name
    per_time_df = build_per_time_decisions(examples_index, y_examples, all_scores, split_by_time)
    split_summary_df = (
        per_time_df.groupby("split", sort=False)
        .agg(
            n_times=("time_id", "count"),
            contains_closest=("contains_closest", "mean"),
            saved_top1=("saved_top1", "mean"),
            saved_top3=("saved_top3", "mean"),
            mean_rank=("saved_rank", "mean"),
            avg_regret=("regret", "mean"),
            avg_norm_regret=("norm_regret", "mean"),
        )
        .reset_index()
    )

    history_df = pd.DataFrame(history)
    metrics_df = pd.DataFrame(
        [
            {
                "run_name": config.run_name,
                "split": split_name,
                "best_epoch": best_epoch,
                **split_metrics,
            }
            for split_name, split_metrics in metrics.items()
        ]
    )

    with torch.no_grad():
        action_embeddings = model.embed_action(torch.from_numpy(a_catalog_std.astype(np.float32)).to(device)).detach().cpu().numpy().astype(np.float32)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "context_dim": int(c_rssi.shape[1]),
            "action_dim": int(a_examples.shape[1]),
            "best_epoch": int(best_epoch),
            "metrics": metrics,
            "standardizers": {
                "C_mu": c_mu,
                "C_sigma": c_sigma,
                "A_mu": a_mu,
                "A_sigma": a_sigma,
            },
            "context_feature_names": context_names,
            "static_action_feature_names": action_names,
        },
        out_dir / f"{MODEL_TAG}_checkpoint.pt",
    )
    np.savez(
        out_dir / "static_action_embeddings.npz",
        action_embeddings=action_embeddings,
        action_vectors=a_catalog,
        action_vectors_std=a_catalog_std,
        subset_key=subset_key,
        static_action_feature_names=np.asarray(action_names, dtype=object),
    )

    per_time_path = table_dir / f"{MODEL_TAG}_per_time_decisions.csv"
    split_summary_path = table_dir / f"{MODEL_TAG}_split_summary.csv"
    history_path = table_dir / f"{MODEL_TAG}_history.csv"
    metrics_path = table_dir / f"{MODEL_TAG}_metrics.csv"
    per_time_df.to_csv(per_time_path, index=False)
    split_summary_df.to_csv(split_summary_path, index=False)
    history_df.to_csv(history_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    test_row = split_summary_df.loc[split_summary_df["split"].eq("test")].iloc[0]
    mean_acc = float(test_row["contains_closest"])
    n_valid = int(test_row["n_times"])
    rssi_acc_row = {
        "method": "RSSI Two-Tower",
        "display_name": "RSSI Two-Tower",
        "n_valid": n_valid,
        "mean_accuracy": mean_acc,
        "std_error_binomial": float(np.sqrt(mean_acc * (1.0 - mean_acc) / n_valid)),
        "n_common_times": n_valid,
        "mean_accuracy_common_times": mean_acc,
        "rolling_window": 100,
        "source_csv": str(per_time_path),
    }

    base_acc_path = table_dir / "all_methods_mean_accuracy_table.csv"
    if base_acc_path.exists():
        base_acc = pd.read_csv(base_acc_path)
        base_acc = base_acc[~base_acc["method"].eq("RSSI Two-Tower")].copy()
        insert_at = 1 if "Two-Tower" in set(base_acc["method"]) else len(base_acc)
        acc_with_rssi = pd.concat(
            [base_acc.iloc[:insert_at], pd.DataFrame([rssi_acc_row]), base_acc.iloc[insert_at:]],
            ignore_index=True,
        )
    else:
        acc_with_rssi = pd.DataFrame([rssi_acc_row])
    acc_with_rssi_path = table_dir / "all_methods_mean_accuracy_table_with_rssi_tower.csv"
    acc_with_rssi_latex_path = table_dir / "all_methods_mean_accuracy_table_with_rssi_tower_latex.txt"
    acc_with_rssi.to_csv(acc_with_rssi_path, index=False)
    acc_with_rssi_latex_path.write_text(
        "\n".join(
            f"{row.display_name} & {100.0 * row.mean_accuracy:.2f} & {100.0 * row.std_error_binomial:.2f} \\\\"
            for row in acc_with_rssi.itertuples(index=False)
        ),
        encoding="utf-8",
    )

    manifest = {
        "run_name": config.run_name,
        "description": "RSSI-only two-tower baseline: context swaps v3 audio bands for per-node rssi_db while keeping context node coordinates and the same static action masks/coordinates.",
        "context_feature_names": context_names,
        "static_action_feature_names": action_names,
        "best_epoch": int(best_epoch),
        "max_epochs": int(config.max_epochs),
        "metrics": metrics,
    }
    with open(out_dir / f"{MODEL_TAG}_info.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Saved RSSI-only two-tower artifacts:")
    print(" ", out_dir)
    print(" ", per_time_path)
    print(" ", acc_with_rssi_path)

    return {
        "model": model,
        "config": config,
        "best_epoch": best_epoch,
        "history": history_df,
        "metrics": metrics_df,
        "split_summary": split_summary_df,
        "per_time": per_time_df,
        "mean_accuracy_with_rssi": acc_with_rssi,
        "paths": {
            "out_dir": out_dir,
            "per_time": per_time_path,
            "split_summary": split_summary_path,
            "history": history_path,
            "metrics": metrics_path,
            "accuracy_with_rssi": acc_with_rssi_path,
        },
    }
