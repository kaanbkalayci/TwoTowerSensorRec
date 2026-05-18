from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    run_name: str = "two_tower_h64_e8"
    utility_name: str = "saved"
    utility_kwargs: dict[str, float] | None = None
    hidden: int = 64
    emb_dim: int = 8
    depth: int = 2
    dropout: float = 0.05
    combine_mode: str = "mul_only"
    loss_name: str = "mse"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    max_epochs: int = 25
    patience: int = 8
    seed: int = 22
    train_frac: float = 0.60
    val_frac: float = 0.20
    log_every: int = 1
    num_workers: int = 0


class PairDataset(Dataset):
    def __init__(
        self,
        C_by_time: np.ndarray,
        A_examples: np.ndarray,
        y_examples: np.ndarray,
        example_time_id: np.ndarray,
        indices: np.ndarray,
    ):
        self.C_by_time = torch.from_numpy(C_by_time.astype(np.float32))
        self.A_examples = torch.from_numpy(A_examples.astype(np.float32))
        self.y_examples = torch.from_numpy(y_examples.astype(np.float32))
        self.example_time_id = torch.from_numpy(example_time_id.astype(np.int64))
        self.indices = torch.from_numpy(indices.astype(np.int64))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        time_id = self.example_time_id[idx]
        return self.C_by_time[time_id], self.A_examples[idx], self.y_examples[idx]


class TwoTowerMLP(nn.Module):
    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        hidden: int = 64,
        emb_dim: int = 8,
        depth: int = 2,
        dropout: float = 0.05,
        combine_mode: str = "mul_only",
    ):
        super().__init__()
        self.combine_mode = combine_mode
        self.context_tower = self._tower(context_dim, hidden, emb_dim, depth, dropout)
        self.action_tower = self._tower(action_dim, hidden, emb_dim, depth, dropout)

        if combine_mode == "dot":
            self.head = None
        elif combine_mode == "mul_only":
            self.head = self._head(emb_dim * 3, hidden, dropout)
        elif combine_mode == "concat_abs":
            self.head = self._head(emb_dim * 4, hidden, dropout)
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")

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
        if self.combine_mode == "dot":
            return (c_emb * a_emb).sum(dim=-1)
        if self.combine_mode == "mul_only":
            rep = torch.cat([c_emb, a_emb, c_emb * a_emb], dim=-1)
        else:
            rep = torch.cat([c_emb, a_emb, c_emb * a_emb, torch.abs(c_emb - a_emb)], dim=-1)
        return self.head(rep).squeeze(-1)

    def forward(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.score_embeddings(self.embed_context(context), self.embed_action(action))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_two_tower_data(
    processed_dir: str | Path,
    prefix: str = "vehicle_sensor_subset_200ms",
    *,
    utility_name: str = "saved",
    utility_kwargs: dict[str, float] | None = None,
    max_time_steps: int | None = None,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    arrays = np.load(processed_dir / f"{prefix}_arrays.npz", allow_pickle=True)
    examples_index = pd.read_csv(processed_dir / f"{prefix}_examples_index.csv")

    with open(processed_dir / f"{prefix}_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    C_by_time = arrays["C_by_time"].astype(np.float32)
    A_examples = arrays["A_examples"].astype(np.float32)
    saved_y = arrays["y_examples"].astype(np.float32)
    example_time_id = arrays["example_time_id"].astype(np.int64)
    sequence_times = arrays["sequence_times"]

    if max_time_steps is not None:
        keep_time = min(int(max_time_steps), len(C_by_time))
        keep_examples = example_time_id < keep_time
        C_by_time = C_by_time[:keep_time]
        sequence_times = sequence_times[:keep_time]
        A_examples = A_examples[keep_examples]
        saved_y = saved_y[keep_examples]
        example_time_id = example_time_id[keep_examples]
        examples_index = examples_index.loc[keep_examples].reset_index(drop=True)

    y_examples = build_utility_labels(
        examples_index=examples_index,
        saved_y=saved_y,
        meta=meta,
        utility_name=utility_name,
        utility_kwargs=utility_kwargs or {},
    ).astype(np.float32)

    return {
        "C_by_time": C_by_time,
        "A_examples": A_examples,
        "y_examples": y_examples,
        "saved_y_examples": saved_y,
        "example_time_id": example_time_id,
        "sequence_times": sequence_times,
        "examples_index": examples_index,
        "meta": meta,
        "utility_name": utility_name,
        "utility_kwargs": utility_kwargs or {},
    }


def build_utility_labels(
    examples_index: pd.DataFrame,
    saved_y: np.ndarray,
    meta: dict[str, Any],
    utility_name: str,
    utility_kwargs: dict[str, float],
) -> np.ndarray:
    if utility_name == "saved":
        return saved_y.copy()

    d1 = examples_index["d1"].to_numpy(dtype=float)
    d2 = examples_index["d2"].to_numpy(dtype=float)
    d3 = examples_index["d3"].to_numpy(dtype=float)
    w2 = float(utility_kwargs.get("utility_second_weight", meta.get("utility_second_weight", 0.45)))
    w3 = float(utility_kwargs.get("utility_third_weight", meta.get("utility_third_weight", 0.20)))

    if utility_name == "rational":
        rho = float(utility_kwargs.get("rho", meta.get("rho", max(np.nanmedian(d1), 1.0))))
        y = 1.0 / (1.0 + d1 / rho)
        y += np.where(np.isfinite(d2), w2 / (1.0 + d2 / rho), 0.0)
        y += np.where(np.isfinite(d3), w3 / (1.0 + d3 / rho), 0.0)
        return y

    if utility_name == "clipped_linear":
        radius = float(utility_kwargs.get("utility_radius", 2.0 * meta.get("rho", max(np.nanmedian(d1), 1.0))))

        def term(d: np.ndarray) -> np.ndarray:
            return np.maximum(0.0, 1.0 - d / max(radius, 1e-8))

        y = term(d1)
        y += np.where(np.isfinite(d2), w2 * term(d2), 0.0)
        y += np.where(np.isfinite(d3), w3 * term(d3), 0.0)
        return y

    if utility_name == "closest_binary":
        return examples_index["contains_closest_node"].to_numpy(dtype=float)

    if utility_name == "rank_discount":
        rank = examples_index["best_rank_in_subset"].to_numpy(dtype=float)
        return 1.0 / np.maximum(rank, 1.0)

    raise ValueError(f"Unknown utility_name: {utility_name}")


def chronological_split(
    example_time_id: np.ndarray,
    n_times: int,
    train_frac: float,
    val_frac: float,
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


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)


def prepare_standardized_data(data: dict[str, Any], config: TrainConfig) -> dict[str, Any]:
    split = chronological_split(
        data["example_time_id"],
        n_times=len(data["C_by_time"]),
        train_frac=config.train_frac,
        val_frac=config.val_frac,
    )

    C_mu, C_sigma = fit_standardizer(data["C_by_time"][split["train_time_ids"]])
    A_mu, A_sigma = fit_standardizer(data["A_examples"][split["train"]])

    prepared = dict(data)
    prepared["C_by_time_std"] = ((data["C_by_time"] - C_mu) / C_sigma).astype(np.float32)
    prepared["A_examples_std"] = ((data["A_examples"] - A_mu) / A_sigma).astype(np.float32)
    prepared["split"] = split
    prepared["standardizers"] = {
        "C_mu": C_mu,
        "C_sigma": C_sigma,
        "A_mu": A_mu,
        "A_sigma": A_sigma,
    }
    return prepared


def make_loss(loss_name: str):
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=0.05)
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss_name: {loss_name}")


def make_loader(prepared: dict[str, Any], indices: np.ndarray, config: TrainConfig, shuffle: bool) -> DataLoader:
    ds = PairDataset(
        prepared["C_by_time_std"],
        prepared["A_examples_std"],
        prepared["y_examples"],
        prepared["example_time_id"],
        indices,
    )
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    prepared: dict[str, Any],
    indices: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    loader = make_loader(
        prepared,
        indices,
        TrainConfig(batch_size=batch_size, num_workers=0),
        shuffle=False,
    )
    model.eval()
    scores = []
    for C, A, _ in loader:
        C = C.to(device)
        A = A.to(device)
        scores.append(model(C, A).detach().cpu().numpy())
    return np.concatenate(scores)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return {"rmse": rmse, "mae": mae, "r2": float(r2)}


def decision_metrics(time_id: np.ndarray, y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Decision-facing ranking metrics.

    For each timestep, the model selects the highest-scored subset. top1 asks
    whether that selected subset is truly optimal. top3 asks whether that
    selected subset is within the three best true-utility actions. mean_rank is
    the average true-utility rank of the selected subset.
    """
    df = pd.DataFrame({"time_id": time_id, "y_true": y_true, "score": scores})
    top1_hits = []
    top3_hits = []
    ranks = []
    regrets = []
    norm_regrets = []

    for _, group in df.groupby("time_id", sort=False):
        yt = group["y_true"].to_numpy(dtype=float)
        sc = group["score"].to_numpy(dtype=float)
        order = np.argsort(-sc)

        best_value = float(np.max(yt))
        worst_value = float(np.min(yt))
        pred_best = int(order[0])
        selected_value = float(yt[pred_best])
        selected_true_rank = int(np.sum(yt > selected_value + 1e-10)) + 1

        top1_hits.append(float(selected_true_rank == 1))
        top3_hits.append(float(selected_true_rank <= 3))
        ranks.append(selected_true_rank)

        regret = best_value - selected_value
        regrets.append(regret)
        denom = best_value - worst_value
        norm_regrets.append(0.0 if denom <= 1e-12 else regret / denom)

    return {
        "top1": float(np.mean(top1_hits)),
        "top3": float(np.mean(top3_hits)),
        "mean_rank": float(np.mean(ranks)),
        "avg_regret": float(np.mean(regrets)),
        "avg_norm_regret": float(np.mean(norm_regrets)),
    }


def evaluate_split(
    model: nn.Module,
    prepared: dict[str, Any],
    split_name: str,
    config: TrainConfig,
    device: str,
) -> dict[str, float]:
    indices = prepared["split"][split_name]
    scores = predict_scores(model, prepared, indices, config.batch_size * 2, device)
    y = prepared["y_examples"][indices]
    time_id = prepared["example_time_id"][indices]
    value_scores = 1.0 / (1.0 + np.exp(-scores)) if config.loss_name == "bce" else scores
    out = {}
    out.update(regression_metrics(y, value_scores))
    out.update(decision_metrics(time_id, y, scores))
    return out


def train_one_config(
    data: dict[str, Any],
    config: TrainConfig,
    *,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    set_all_seeds(config.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if data.get("utility_name") != config.utility_name or data.get("utility_kwargs") != (config.utility_kwargs or {}):
        data = dict(data)
        data["y_examples"] = build_utility_labels(
            data["examples_index"],
            data["saved_y_examples"],
            data["meta"],
            config.utility_name,
            config.utility_kwargs or {},
        ).astype(np.float32)
        data["utility_name"] = config.utility_name
        data["utility_kwargs"] = config.utility_kwargs or {}

    prepared = prepare_standardized_data(data, config)
    train_loader = make_loader(prepared, prepared["split"]["train"], config, shuffle=True)
    model = TwoTowerMLP(
        context_dim=prepared["C_by_time"].shape[1],
        action_dim=prepared["A_examples"].shape[1],
        hidden=config.hidden,
        emb_dim=config.emb_dim,
        depth=config.depth,
        dropout=config.dropout,
        combine_mode=config.combine_mode,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = make_loss(config.loss_name)

    best_state = None
    best_epoch = -1
    best_val_key = (np.inf, np.inf)
    wait = 0
    history = []
    start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for C, A, y in train_loader:
            C = C.to(device)
            A = A.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(C, A)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(losses))
        val = evaluate_split(model, prepared, "val", config, device)
        val_key = (val["avg_regret"], val["rmse"])
        improved = val_key < best_val_key

        if improved:
            best_val_key = val_key
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)

        should_log = verbose and (epoch == 1 or epoch % config.log_every == 0 or improved or epoch == config.max_epochs)
        if should_log:
            star = "*" if improved else " "
            elapsed = time.time() - start
            print(
                f"{config.run_name:>24s} ep {epoch:03d}/{config.max_epochs:03d}{star} "
                f"loss={train_loss:.5f} val_rmse={val['rmse']:.4f} "
                f"top1={val['top1']:.3f} top3={val['top3']:.3f} "
                f"reg={val['avg_regret']:.4f} rank={val['mean_rank']:.2f} "
                f"{elapsed:.0f}s"
            )

        if wait >= config.patience:
            if verbose:
                print(f"{config.run_name:>24s} early stop at ep {epoch:03d}; best ep {best_epoch:03d}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        split: evaluate_split(model, prepared, split, config, device)
        for split in ("train", "val", "test")
    }

    return {
        "config": config,
        "model": model,
        "prepared": prepared,
        "history": pd.DataFrame(history),
        "metrics": metrics,
        "best_epoch": best_epoch,
        "device": device,
    }


def run_config_grid(
    processed_dir: str | Path,
    configs: list[TrainConfig],
    *,
    prefix: str = "vehicle_sensor_subset_200ms",
    max_time_steps: int | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    results = []
    cache: dict[tuple[str, str], dict[str, Any]] = {}

    for config in configs:
        utility_key = json.dumps(config.utility_kwargs or {}, sort_keys=True)
        key = (config.utility_name, utility_key)
        if key not in cache:
            cache[key] = load_processed_two_tower_data(
                processed_dir,
                prefix=prefix,
                utility_name=config.utility_name,
                utility_kwargs=config.utility_kwargs,
                max_time_steps=max_time_steps,
            )
        results.append(train_one_config(cache[key], config, device=device, verbose=True))
    return results


def summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        cfg = result["config"]
        row = {
            "run_name": cfg.run_name,
            "utility": cfg.utility_name,
            "hidden": cfg.hidden,
            "emb_dim": cfg.emb_dim,
            "depth": cfg.depth,
            "dropout": cfg.dropout,
            "combine": cfg.combine_mode,
            "loss": cfg.loss_name,
            "best_epoch": result["best_epoch"],
        }
        for split, metrics in result["metrics"].items():
            for key, value in metrics.items():
                row[f"{split}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["val_avg_regret", "val_rmse"]).reset_index(drop=True)


def evaluate_result_on_utility(
    result: dict[str, Any],
    utility_name: str,
    utility_kwargs: dict[str, float] | None = None,
    *,
    split_names: tuple[str, ...] = ("val", "test"),
) -> dict[str, dict[str, float]]:
    """
    Score one trained model against a common evaluation utility.

    This is for comparing different training utilities as decision policies.
    The model scores are left on their native scale; metrics are rank/regret
    metrics computed against the requested evaluation utility labels.
    """
    model = result["model"]
    prepared = result["prepared"]
    config: TrainConfig = result["config"]
    device = result["device"]

    eval_y = build_utility_labels(
        examples_index=prepared["examples_index"],
        saved_y=prepared["saved_y_examples"],
        meta=prepared["meta"],
        utility_name=utility_name,
        utility_kwargs=utility_kwargs or {},
    ).astype(np.float32)

    out = {}
    for split_name in split_names:
        indices = prepared["split"][split_name]
        scores = predict_scores(model, prepared, indices, config.batch_size * 2, device)
        out[split_name] = decision_metrics(
            prepared["example_time_id"][indices],
            eval_y[indices],
            scores,
        )
    return out


def compare_on_common_objectives(
    results: list[dict[str, Any]],
    objectives: list[dict[str, Any]] | None = None,
    *,
    split_names: tuple[str, ...] = ("val", "test"),
) -> pd.DataFrame:
    """
    Compare trained runs on shared downstream objectives.

    Each row is one trained run evaluated against one objective on one split.
    This makes utility-function experiments comparable even when the training
    labels differ.
    """
    if objectives is None:
        objectives = [
            {"name": "contains_closest", "utility_name": "closest_binary", "utility_kwargs": {}},
            {"name": "rank_discount", "utility_name": "rank_discount", "utility_kwargs": {}},
            {"name": "saved_rational", "utility_name": "saved", "utility_kwargs": {}},
        ]

    rows = []
    for result in results:
        cfg: TrainConfig = result["config"]
        for objective in objectives:
            objective_metrics = evaluate_result_on_utility(
                result,
                utility_name=objective["utility_name"],
                utility_kwargs=objective.get("utility_kwargs") or {},
                split_names=split_names,
            )
            for split_name, metrics in objective_metrics.items():
                rows.append(
                    {
                        "run_name": cfg.run_name,
                        "train_utility": cfg.utility_name,
                        "eval_objective": objective["name"],
                        "split": split_name,
                        "hidden": cfg.hidden,
                        "emb_dim": cfg.emb_dim,
                        "combine": cfg.combine_mode,
                        **metrics,
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(["eval_objective", "split", "avg_norm_regret", "mean_rank"])
        .reset_index(drop=True)
    )


@torch.no_grad()
def export_frozen_embeddings(
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    prefix: str | None = None,
) -> dict[str, Path]:
    model: TwoTowerMLP = result["model"]
    prepared = result["prepared"]
    config: TrainConfig = result["config"]
    device = result["device"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or config.run_name

    model.eval()
    C = torch.from_numpy(prepared["C_by_time_std"].astype(np.float32)).to(device)
    context_emb = model.embed_context(C).cpu().numpy().astype(np.float32)

    action_emb_batches = []
    scores = []
    A = prepared["A_examples_std"]
    time_id = prepared["example_time_id"]
    batch_size = config.batch_size * 2

    for start in range(0, len(A), batch_size):
        end = min(start + batch_size, len(A))
        A_batch = torch.from_numpy(A[start:end].astype(np.float32)).to(device)
        C_batch = torch.from_numpy(prepared["C_by_time_std"][time_id[start:end]].astype(np.float32)).to(device)
        c_emb = model.embed_context(C_batch)
        a_emb = model.embed_action(A_batch)
        action_emb_batches.append(a_emb.cpu().numpy().astype(np.float32))
        scores.append(model.score_embeddings(c_emb, a_emb).cpu().numpy().astype(np.float32))

    paths = {
        "checkpoint": output_dir / f"{prefix}_checkpoint.pt",
        "embeddings": output_dir / f"{prefix}_embeddings.npz",
        "history": output_dir / f"{prefix}_history.csv",
        "metrics": output_dir / f"{prefix}_metrics.json",
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "context_dim": int(prepared["C_by_time"].shape[1]),
            "action_dim": int(prepared["A_examples"].shape[1]),
            "standardizers": prepared["standardizers"],
            "meta": prepared["meta"],
        },
        paths["checkpoint"],
    )
    np.savez_compressed(
        paths["embeddings"],
        context_embeddings=context_emb,
        action_embeddings=np.concatenate(action_emb_batches, axis=0),
        scores=np.concatenate(scores, axis=0),
        example_time_id=time_id,
        y_examples=prepared["y_examples"],
        sequence_times=prepared["sequence_times"],
    )
    result["history"].to_csv(paths["history"], index=False)
    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)

    return paths
