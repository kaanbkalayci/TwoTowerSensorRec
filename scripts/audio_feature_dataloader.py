import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from shapely.geometry import Point


EPS = np.finfo(np.float32).eps

FEATURE_COLUMNS = [
    "rms_db",
    "peak_db",
    "crest_factor_db",
    "zcr",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff85_hz",
    "spectral_flatness",
    "spectral_entropy",
    "band_20_120_db",
    "band_120_500_db",
    "band_500_2000_db",
    "band_2000_6000_db",
    "band_20_120_ratio",
    "band_120_500_ratio",
    "band_500_2000_ratio",
    "band_2000_6000_ratio",
    "low_to_voice_db",
    "mid_to_voice_db",
]

CROSS_SENSOR_FEATURES = [
    "rms_db",
    "spectral_centroid_hz",
    "spectral_flatness",
    "band_20_120_ratio",
    "band_120_500_ratio",
    "band_500_2000_ratio",
    "low_to_voice_db",
    "mid_to_voice_db",
]


def _pick_utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def _load_sensor_nodes(sensor_csv: str | Path, node_list: list[int]) -> gpd.GeoDataFrame:
    nodes = pd.read_csv(sensor_csv).copy()
    if "Node #" in nodes.columns:
        nodes = nodes[nodes["Node #"].isin(node_list)].copy()

    nodes["geometry"] = [Point(lon, lat) for lat, lon in zip(nodes["Lat"], nodes["Lon"])]
    gdf_nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")

    local_epsg = _pick_utm_epsg(float(gdf_nodes["Lat"].mean()), float(gdf_nodes["Lon"].mean()))
    return gdf_nodes.to_crs(epsg=local_epsg)


def _parse_flac_start(file_path: Path, node_id: int) -> datetime | None:
    match = re.match(
        rf"(\d{{8}})_(\d{{6}})_.*_{node_id}_respeaker\.flac$",
        file_path.name,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y%m%d %H%M%S")


def _find_node_flacs(data_dir: str | Path, node_id: int) -> list[Path]:
    data_dir = Path(data_dir)
    pattern = re.compile(rf"\d{{8}}_\d{{6}}_.*_{node_id}_respeaker\.flac$", re.IGNORECASE)
    return sorted(path for path in data_dir.iterdir() if pattern.match(path.name))


def _to_respeaker_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)

    hi = min(audio.shape[1], 5)
    lo = 1 if audio.shape[1] > 1 else 0
    return audio[:, lo:hi].mean(axis=1).astype(np.float32, copy=False)


def _band_power(power: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return np.full(power.shape[0], np.nan, dtype=np.float32)
    return power[:, mask].sum(axis=1)


def _frame_features(audio: np.ndarray, samplerate: int, sample_ms: int) -> pd.DataFrame:
    frame_len = int(round((sample_ms / 1000.0) * samplerate))
    if frame_len <= 0:
        raise ValueError("sample_ms must produce at least one sample per frame.")

    num_frames = len(audio) // frame_len
    if num_frames == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    frames = audio[: num_frames * frame_len].reshape(num_frames, frame_len)
    centered = frames - frames.mean(axis=1, keepdims=True)

    rms = np.sqrt(np.mean(centered**2, axis=1) + EPS)
    peak = np.max(np.abs(centered), axis=1) + EPS
    signs = np.signbit(centered)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

    window = np.hanning(frame_len).astype(np.float32)
    spectrum = np.fft.rfft(centered * window, axis=1)
    power = np.abs(spectrum).astype(np.float32) ** 2
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / samplerate).astype(np.float32)
    total_power = power.sum(axis=1) + EPS
    prob = power / total_power[:, None]

    centroid = (power * freqs[None, :]).sum(axis=1) / total_power
    bandwidth = np.sqrt(
        (power * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total_power
    )

    cumulative = np.cumsum(power, axis=1)
    rolloff_idx = np.argmax(cumulative >= (0.85 * total_power)[:, None], axis=1)
    rolloff = freqs[rolloff_idx]

    flatness = np.exp(np.mean(np.log(power + EPS), axis=1)) / (np.mean(power + EPS, axis=1))
    entropy = -(prob * np.log2(prob + EPS)).sum(axis=1) / np.log2(power.shape[1])

    band_20_120 = _band_power(power, freqs, 20, 120)
    band_120_500 = _band_power(power, freqs, 120, 500)
    band_500_2000 = _band_power(power, freqs, 500, 2000)
    band_2000_6000 = _band_power(power, freqs, 2000, 6000)

    voice_power = band_500_2000 + EPS
    features = pd.DataFrame(
        {
            "rms_db": 20 * np.log10(rms + EPS),
            "peak_db": 20 * np.log10(peak),
            "crest_factor_db": 20 * np.log10(peak / (rms + EPS)),
            "zcr": zcr,
            "spectral_centroid_hz": centroid,
            "spectral_bandwidth_hz": bandwidth,
            "spectral_rolloff85_hz": rolloff,
            "spectral_flatness": flatness,
            "spectral_entropy": entropy,
            "band_20_120_db": 10 * np.log10(band_20_120 + EPS),
            "band_120_500_db": 10 * np.log10(band_120_500 + EPS),
            "band_500_2000_db": 10 * np.log10(band_500_2000 + EPS),
            "band_2000_6000_db": 10 * np.log10(band_2000_6000 + EPS),
            "band_20_120_ratio": band_20_120 / total_power,
            "band_120_500_ratio": band_120_500 / total_power,
            "band_500_2000_ratio": band_500_2000 / total_power,
            "band_2000_6000_ratio": band_2000_6000 / total_power,
            "low_to_voice_db": 10 * np.log10((band_20_120 + EPS) / voice_power),
            "mid_to_voice_db": 10 * np.log10((band_120_500 + EPS) / voice_power),
        }
    )

    features["rms_delta_db"] = features["rms_db"].diff().fillna(0.0)
    features["centroid_delta_hz"] = features["spectral_centroid_hz"].diff().fillna(0.0)
    features["flatness_delta"] = features["spectral_flatness"].diff().fillna(0.0)
    return features


def _load_node_audio_features(
    data_dir: str | Path,
    node_id: int,
    sample_ms: int,
) -> pd.DataFrame | None:
    node_frames = []
    for file_path in _find_node_flacs(data_dir, node_id):
        start_time = _parse_flac_start(file_path, node_id)
        if start_time is None:
            continue

        audio, samplerate = sf.read(file_path, dtype="float32")
        mono = _to_respeaker_mono(audio)
        features = _frame_features(mono, samplerate=samplerate, sample_ms=sample_ms)
        if features.empty:
            print(f"Audio too short for node {node_id} in {file_path.name}; skipping.")
            continue

        features.index = pd.date_range(
            start=start_time,
            periods=len(features),
            freq=f"{sample_ms}ms",
            name="datetime",
        )
        node_frames.append(features)
        print(
            f"Loaded audio features: node {node_id}, {file_path.name}, "
            f"{len(features)} rows at {sample_ms} ms"
        )

    if not node_frames:
        print(f"No FLAC files found for node {node_id} in {data_dir}")
        return None

    node_df = pd.concat(node_frames).sort_index()
    return node_df[~node_df.index.duplicated(keep="first")]


def _add_cross_sensor_summaries(
    feature_df: pd.DataFrame,
    node_list: list[int],
    feature_names: list[str],
) -> pd.DataFrame:
    out = feature_df.copy()
    for feature_name in feature_names:
        columns = [f"n{node}_{feature_name}" for node in node_list if f"n{node}_{feature_name}" in out.columns]
        if len(columns) < 2:
            continue

        values = out[columns]
        out[f"mean_{feature_name}"] = values.mean(axis=1)
        out[f"std_{feature_name}"] = values.std(axis=1)
        out[f"range_{feature_name}"] = values.max(axis=1) - values.min(axis=1)
        out[f"argmax_node_{feature_name}"] = values.idxmax(axis=1).str.extract(r"n(\d+)_")[0].astype(float)
    return out


def build_audio_feature_dataset(
    data_dir: str | Path,
    sensor_csv: str | Path,
    node_list: list[int] | None = None,
    sample_ms: int = 200,
    add_cross_sensor: bool = True,
):
    """
    Build an audio-only feature table synchronized at sample_ms.

    This loader intentionally does not read vehicle GPS and does not create
    vehicle latitude, longitude, or distance-to-vehicle columns. Sensor GPS is
    returned separately as static metadata for plotting or sensor-aware models.

    Returns:
        feature_df: wide audio feature table indexed by datetime
        node_feature_df: long feature table with one row per datetime/node
        gdf_nodes: projected sensor metadata
    """
    sensor_nodes = pd.read_csv(sensor_csv)
    if node_list is None:
        node_list = sorted(sensor_nodes["Node #"].astype(int).tolist())
    node_list = [int(node) for node in node_list]

    gdf_nodes = _load_sensor_nodes(sensor_csv, node_list=node_list)

    wide_frames = []
    long_frames = []
    for node in node_list:
        node_features = _load_node_audio_features(data_dir, node_id=node, sample_ms=sample_ms)
        if node_features is None:
            continue

        wide_frames.append(node_features.add_prefix(f"n{node}_"))
        long_node = node_features.copy()
        long_node["node"] = node
        long_frames.append(long_node.reset_index())

    if not wide_frames:
        raise FileNotFoundError(f"No audio features could be extracted from {data_dir}.")

    feature_df = pd.concat(wide_frames, axis=1).sort_index()
    feature_df = feature_df[~feature_df.index.duplicated(keep="first")]

    if add_cross_sensor:
        feature_df = _add_cross_sensor_summaries(
            feature_df,
            node_list=node_list,
            feature_names=CROSS_SENSOR_FEATURES,
        )

    node_feature_df = pd.concat(long_frames, ignore_index=True).sort_values(["datetime", "node"])
    return feature_df, node_feature_df, gdf_nodes


def _time_axis_labels(index: pd.Index, max_ticks: int = 5):
    if len(index) == 0:
        return [], []

    tick_positions = np.linspace(0, len(index) - 1, num=min(max_ticks, len(index)), dtype=int)
    tick_labels = [pd.Timestamp(index[pos]).strftime("%H:%M:%S") for pos in tick_positions]
    return tick_positions, tick_labels


def _plot_feature_heatmap(
    ax,
    node_feature_df: pd.DataFrame,
    node_list: list[int],
    feature_name: str,
    title: str,
    cmap: str,
):
    pivot = node_feature_df.pivot(index="datetime", columns="node", values=feature_name)
    pivot = pivot.reindex(columns=node_list)
    matrix = pivot.T.to_numpy(dtype=float)

    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_yticks(np.arange(len(node_list)))
    ax.set_yticklabels([str(node) for node in node_list])
    ax.set_ylabel("Node")

    ticks, labels = _time_axis_labels(pivot.index)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Time")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def plot_audio_feature_overview(
    feature_df: pd.DataFrame,
    node_feature_df: pd.DataFrame,
    gdf_nodes: gpd.GeoDataFrame,
    node_list: list[int],
    sample_ms: int = 200,
):
    """
    Compact diagnostics for the audio-only feature dataset.
    """
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.0, 1.0])

    ax_nodes = fig.add_subplot(gs[0, 0])
    gdf_nodes.plot(ax=ax_nodes, color="#dc2626", marker="^", markersize=90)
    for _, row in gdf_nodes.iterrows():
        node_id = row.get("Node #", row.get("Node", ""))
        ax_nodes.annotate(
            str(node_id),
            (row.geometry.x, row.geometry.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax_nodes.set_title("Sensor layout only")
    ax_nodes.set_xlabel("Easting (m)")
    ax_nodes.set_ylabel("Northing (m)")
    ax_nodes.grid(True, alpha=0.25)
    ax_nodes.set_aspect("equal", adjustable="datalim")

    ax_gap = fig.add_subplot(gs[0, 1])
    diffs = feature_df.index.to_series().diff().dt.total_seconds().mul(1000)
    ax_gap.hist(diffs.dropna(), bins=30, color="#2563eb", alpha=0.8)
    ax_gap.axvline(sample_ms, color="#dc2626", linestyle="--", linewidth=1)
    ax_gap.set_title(f"Feature index spacing ({sample_ms} ms target)")
    ax_gap.set_xlabel("Delta time (ms)")
    ax_gap.set_ylabel("Count")
    ax_gap.grid(True, alpha=0.25)

    ax_corr = fig.add_subplot(gs[0, 2])
    summary_cols = [
        column
        for column in feature_df.columns
        if column.startswith("mean_") and feature_df[column].notna().any()
    ]
    corr_cols = summary_cols[:8]
    if len(corr_cols) >= 2:
        corr = feature_df[corr_cols].corr().fillna(0.0)
        im = ax_corr.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        labels = [col.replace("mean_", "") for col in corr_cols]
        ax_corr.set_xticks(np.arange(len(labels)))
        ax_corr.set_yticks(np.arange(len(labels)))
        ax_corr.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax_corr.set_yticklabels(labels, fontsize=8)
        ax_corr.set_title("Mean feature correlation")
        plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.02)
    else:
        ax_corr.axis("off")

    heatmaps = [
        ("rms_db", "RMS level (dB)", "magma"),
        ("spectral_centroid_hz", "Spectral centroid (Hz)", "viridis"),
        ("band_20_120_ratio", "20-120 Hz ratio", "cividis"),
        ("band_120_500_ratio", "120-500 Hz ratio", "cividis"),
        ("band_500_2000_ratio", "500-2000 Hz ratio", "cividis"),
        ("spectral_flatness", "Spectral flatness", "plasma"),
    ]

    for idx, (feature_name, title, cmap) in enumerate(heatmaps):
        ax = fig.add_subplot(gs[1 + idx // 3, idx % 3])
        _plot_feature_heatmap(
            ax,
            node_feature_df=node_feature_df,
            node_list=node_list,
            feature_name=feature_name,
            title=title,
            cmap=cmap,
        )

    fig.suptitle("Audio-only feature overview: no vehicle GPS or distance columns", fontsize=14)
    return fig

