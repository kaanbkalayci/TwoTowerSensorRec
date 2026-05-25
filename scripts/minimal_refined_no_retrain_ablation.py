from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from IPython.display import display
except Exception:  # pragma: no cover - notebook convenience
    display = print


PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_tower_training import (  # noqa: E402
    TrainConfig,
    TwoTowerMLP,
    build_utility_labels,
    chronological_split,
    decision_metrics,
    predict_scores,
    regression_metrics,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUN_DIR = PROJECT_ROOT / "experiments" / "feature_search_two_tower" / "fs_minimal_deploy_refined_h512_d2_e16"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "feature_ablation_two_tower"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "feature_ablation_two_tower"
for path in [TABLE_DIR, FIG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

PREFIX = "vehicle_sensor_subset_200ms_expanded_features"
ARRAYS_PATH = PROCESSED_DIR / f"{PREFIX}_arrays.npz"
META_PATH = PROCESSED_DIR / f"{PREFIX}_meta.json"
EXAMPLES_PATH = PROCESSED_DIR / f"{PREFIX}_examples_index.csv"
REFINED_CACHE = PROCESSED_DIR / "vehicle_sensor_subset_200ms_refined_audio_bands_long.csv"
CHECKPOINT_PATH = RUN_DIR / "selected_checkpoint.pt"
MANIFEST_PATH = RUN_DIR / "feature_manifest.json"
STATIC_ACTION_PATH = RUN_DIR / "static_action_embeddings.npz"

for path in [
    ARRAYS_PATH,
    META_PATH,
    EXAMPLES_PATH,
    REFINED_CACHE,
    CHECKPOINT_PATH,
    MANIFEST_PATH,
    STATIC_ACTION_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DEVICE:", DEVICE)
print("RUN_DIR:", RUN_DIR)


def torch_load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


arrays = np.load(ARRAYS_PATH, allow_pickle=True)
with open(META_PATH, "r", encoding="utf-8") as f:
    base_meta = json.load(f)
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

checkpoint = torch_load_checkpoint(CHECKPOINT_PATH)
config = TrainConfig(**checkpoint["config"])

examples_index = pd.read_csv(EXAMPLES_PATH)
examples_index["subset_str"] = examples_index["subset_str"].astype(str)
sequence_times = pd.to_datetime(arrays["sequence_times"])
example_time_id = arrays["example_time_id"].astype(np.int64)
saved_y = arrays["y_examples"].astype(np.float32)
contains_y = build_utility_labels(
    examples_index=examples_index,
    saved_y=saved_y,
    meta=base_meta,
    utility_name="closest_binary",
    utility_kwargs={},
).astype(np.float32)

base_context = arrays["C_by_time"].astype(np.float32)
base_context_names = list(base_meta["context_feature_names"])
base_context_index = {name: i for i, name in enumerate(base_context_names)}

context_names = list(manifest["context_feature_names"])
action_names = list(manifest["static_action_feature_names"])
static_npz = np.load(STATIC_ACTION_PATH, allow_pickle=True)
static_action_names = [str(x) for x in static_npz["static_action_feature_names"]]
if action_names != static_action_names:
    raise RuntimeError("Manifest action features do not match saved static action catalog.")
action_catalog = static_npz["action_vectors"].astype(np.float32)
subset_key = np.asarray(static_npz["subset_key"]).astype(str)
subset_to_action_idx = {name: i for i, name in enumerate(subset_key)}
missing_subset_labels = set(examples_index["subset_str"].unique()) - set(subset_to_action_idx)
if missing_subset_labels:
    raise RuntimeError(f"Subset labels missing from action catalog: {sorted(missing_subset_labels)[:5]}")
example_action_idx = examples_index["subset_str"].map(subset_to_action_idx).to_numpy(dtype=np.int64)

refined = pd.read_csv(REFINED_CACHE)
refined["datetime"] = pd.to_datetime(refined["datetime"])
refined["node"] = refined["node"].astype(int)
sequence_index = pd.DatetimeIndex(sequence_times)
refined_by_node = {
    int(node): node_df.set_index("datetime").sort_index()
    for node, node_df in refined.groupby("node", sort=False)
}


def refined_column(name: str) -> np.ndarray:
    match = re.match(r"^n(\d+)_audio_(.+)$", name)
    if match is None:
        raise KeyError(f"{name!r} is neither a base context feature nor a refined audio feature")
    node = int(match.group(1))
    feature = match.group(2)
    if node not in refined_by_node:
        raise KeyError(f"Missing refined audio for node {node}")
    node_df = refined_by_node[node]
    if feature not in node_df.columns:
        raise KeyError(f"Missing refined audio feature {feature!r} for node {node}")
    values = node_df[feature].reindex(sequence_index).to_numpy(dtype=float)
    if np.isfinite(values).any():
        fill = float(np.nanmedian(values[np.isfinite(values)]))
    else:
        fill = 0.0
    return np.nan_to_num(values, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def build_context_matrix(names: list[str]) -> np.ndarray:
    cols = []
    for name in names:
        if name in base_context_index:
            cols.append(base_context[:, base_context_index[name]].astype(np.float32))
        else:
            cols.append(refined_column(name))
    return np.column_stack(cols).astype(np.float32)


C_raw = build_context_matrix(context_names)
A_raw = action_catalog[example_action_idx].astype(np.float32)
if C_raw.shape[1] != int(checkpoint["context_dim"]):
    raise ValueError((C_raw.shape, checkpoint["context_dim"]))
if A_raw.shape[1] != int(checkpoint["action_dim"]):
    raise ValueError((A_raw.shape, checkpoint["action_dim"]))

std = checkpoint["standardizers"]
C_mu = std["C_mu"].astype(np.float32)
C_sigma = np.where(np.abs(std["C_sigma"]) < 1e-8, 1.0, std["C_sigma"]).astype(np.float32)
A_mu = std["A_mu"].astype(np.float32)
A_sigma = np.where(np.abs(std["A_sigma"]) < 1e-8, 1.0, std["A_sigma"]).astype(np.float32)
C_std_base = ((C_raw - C_mu) / C_sigma).astype(np.float32)
A_std_base = ((A_raw - A_mu) / A_sigma).astype(np.float32)

split = chronological_split(
    example_time_id,
    n_times=len(C_raw),
    train_frac=config.train_frac,
    val_frac=config.val_frac,
)

model = TwoTowerMLP(
    context_dim=C_raw.shape[1],
    action_dim=A_raw.shape[1],
    hidden=config.hidden,
    emb_dim=config.emb_dim,
    depth=config.depth,
    dropout=config.dropout,
    combine_mode=config.combine_mode,
).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def prepared_with(C_std: np.ndarray, A_std: np.ndarray, y: np.ndarray) -> dict:
    return {
        "C_by_time_std": C_std,
        "A_examples_std": A_std,
        "y_examples": y,
        "example_time_id": example_time_id,
        "split": split,
    }


def evaluate_ablation(
    label: str,
    tower: str,
    context_idx: list[int] | None = None,
    action_idx: list[int] | None = None,
) -> dict:
    C_std = C_std_base.copy()
    A_std = A_std_base.copy()
    context_idx = sorted(set(context_idx or []))
    action_idx = sorted(set(action_idx or []))
    if context_idx:
        C_std[:, context_idx] = 0.0
    if action_idx:
        A_std[:, action_idx] = 0.0

    prepared = prepared_with(C_std, A_std, saved_y)
    test_idx = split["test"]
    scores = predict_scores(model, prepared, test_idx, config.batch_size * 2, DEVICE)
    saved_metrics = {
        **regression_metrics(saved_y[test_idx], scores),
        **decision_metrics(example_time_id[test_idx], saved_y[test_idx], scores),
    }
    contains_metrics = decision_metrics(example_time_id[test_idx], contains_y[test_idx], scores)
    return {
        "ablation": label,
        "tower": tower,
        "context_features_masked": len(context_idx),
        "action_features_masked": len(action_idx),
        "test_contains_closest": contains_metrics["top1"],
        "test_saved_top1": saved_metrics["top1"],
        "test_saved_top3": saved_metrics["top3"],
        "test_mean_rank": saved_metrics["mean_rank"],
        "test_avg_regret": saved_metrics["avg_regret"],
        "test_avg_norm_regret": saved_metrics["avg_norm_regret"],
        "test_rmse": saved_metrics["rmse"],
    }


def cidx(pattern: str) -> list[int]:
    rx = re.compile(pattern)
    return [i for i, name in enumerate(context_names) if rx.search(name)]


def aidx(pattern: str) -> list[int]:
    rx = re.compile(pattern)
    return [i for i, name in enumerate(action_names) if rx.search(name)]


band_pairs = [
    ("band_20_80", r"band_20_80_(db|ratio)$"),
    ("band_80_160", r"band_80_160_(db|ratio)$"),
    ("band_160_400", r"band_160_400_(db|ratio)$"),
    ("band_400_900", r"band_400_900_(db|ratio)$"),
    ("band_900_2000", r"band_900_2000_(db|ratio)$"),
    ("band_2000_3500", r"band_2000_3500_(db|ratio)$"),
    ("band_3500_6000", r"band_3500_6000_(db|ratio)$"),
]

ablation_specs = [
    ("baseline_no_mask", "none", [], []),
    ("ctx_all_rssi_rank_share", "context", cidx(r"(rssi_persistence_top3|rssi_persistence_top5|energy_share|rank_percentile)$"), []),
    ("ctx_rssi_persistence_top3", "context", cidx(r"rssi_persistence_top3$"), []),
    ("ctx_rssi_persistence_top5", "context", cidx(r"rssi_persistence_top5$"), []),
    ("ctx_energy_share", "context", cidx(r"energy_share$"), []),
    ("ctx_rank_percentile", "context", cidx(r"rank_percentile$"), []),
    ("ctx_sensor_geometry", "context", cidx(r"sensor_[xy]_norm$"), []),
    ("ctx_spectral_core", "context", cidx(r"(spectral_flatness|spectral_entropy|zcr)$"), []),
    ("ctx_all_refined_bands", "context", cidx(r"band_\d+_\d+_(db|ratio)$"), []),
    ("ctx_refined_db_only", "context", cidx(r"band_\d+_\d+_db$"), []),
    ("ctx_refined_ratio_only", "context", cidx(r"band_\d+_\d+_ratio$"), []),
    ("ctx_refined_low_mid_20_2000", "context", cidx(r"band_(20_80|80_160|160_400|400_900|900_2000)_(db|ratio)$"), []),
    ("ctx_refined_high_2000_6000", "context", cidx(r"band_(2000_3500|3500_6000)_(db|ratio)$"), []),
    ("act_masks", "action", [], aidx(r"^mask_n")),
    ("act_slot_sensor_geometry", "action", [], aidx(r"^slot\d+_sensor_[xy]_norm$")),
    ("act_subset_size", "action", [], aidx(r"^subset_size$")),
]
ablation_specs.extend((f"ctx_{label}", "context", cidx(pattern), []) for label, pattern in band_pairs)

rows = []
for label, tower, context_idx, action_idx in ablation_specs:
    row = evaluate_ablation(label, tower, context_idx, action_idx)
    rows.append(row)
    print(
        f"{label}: contains={row['test_contains_closest']:.4f} "
        f"top3={row['test_saved_top3']:.4f} rank={row['test_mean_rank']:.2f}"
    )

ablation_df = pd.DataFrame(rows)
baseline = ablation_df.loc[ablation_df["ablation"].eq("baseline_no_mask")].iloc[0]
ablation_df["contains_drop_pp"] = 100.0 * (
    float(baseline["test_contains_closest"]) - ablation_df["test_contains_closest"]
)
ablation_df["saved_top3_drop_pp"] = 100.0 * (
    float(baseline["test_saved_top3"]) - ablation_df["test_saved_top3"]
)
ablation_df["mean_rank_increase"] = ablation_df["test_mean_rank"] - float(baseline["test_mean_rank"])
ablation_df["avg_regret_increase"] = ablation_df["test_avg_regret"] - float(baseline["test_avg_regret"])
ablation_df = ablation_df.sort_values(
    ["contains_drop_pp", "saved_top3_drop_pp", "mean_rank_increase"],
    ascending=[False, False, False],
).reset_index(drop=True)

csv_path = TABLE_DIR / "minimal_refined_no_retrain_feature_ablation.csv"
ablation_df.to_csv(csv_path, index=False)

plot_df = ablation_df.loc[~ablation_df["ablation"].eq("baseline_no_mask")].head(16).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.2, 5.2))
colors = np.where(plot_df["contains_drop_pp"] >= 0.0, "#2f6f9f", "#8a8a8a")
ax.barh(plot_df["ablation"], plot_df["contains_drop_pp"], color=colors)
ax.axvline(0.0, color="0.25", linewidth=0.8)
ax.set_xlabel("Closest-containment drop after mean masking (percentage points)")
ax.set_title("No-retrain feature ablation of minimal refined two-tower")
ax.grid(axis="x", alpha=0.25)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "minimal_refined_no_retrain_feature_ablation.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "minimal_refined_no_retrain_feature_ablation.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved:")
print(" ", csv_path)
print(" ", FIG_DIR / "minimal_refined_no_retrain_feature_ablation.pdf")
display(ablation_df)
