import os
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from shapely.geometry import Point


def _pick_utm_epsg(lat: float, lon: float) -> int:
    """
    Pick a local UTM CRS from a latitude/longitude pair.
    """
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def _to_datetime_index_from_first_column(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Robust datetime parsing for GPS CSVs.
    Assumes the first column is the timestamp column.
    """
    first_col = df.iloc[:, 0]

    # First try direct parse.
    parsed = pd.to_datetime(first_col, errors="coerce")

    # If that fails heavily, try minute-second formats used in older pipelines.
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(first_col, format="%M:%S.%f", errors="coerce")

    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(first_col, format="%M:%S", errors="coerce")

    if parsed.isna().all():
        raise ValueError("Could not parse datetime column in GPS CSV.")

    return parsed


def _read_vehicle_gps(gps_csv: str) -> pd.DataFrame:
    """
    Reads vehicle GPS CSV and returns a DataFrame indexed by datetime
    with columns Latitude and Longitude.
    """
    gps_df = pd.read_csv(gps_csv)
    dt_index = _to_datetime_index_from_first_column(gps_df)
    gps_df = gps_df.copy()
    gps_df["datetime"] = dt_index
    gps_df = gps_df.dropna(subset=["datetime"]).set_index("datetime")

    if gps_df.shape[1] < 3:
        raise ValueError(
            f"GPS CSV {gps_csv} must have at least 3 columns: time, lat, lon."
        )

    latitude_longitude_df = gps_df.iloc[:, 1:3].copy()
    latitude_longitude_df.columns = ["Latitude", "Longitude"]

    # Drop duplicated timestamps, sort.
    latitude_longitude_df = latitude_longitude_df[
        ~latitude_longitude_df.index.duplicated(keep="first")
    ].sort_index()

    return latitude_longitude_df


def _resample_vehicle_gps(
    latitude_longitude_df: pd.DataFrame,
    resample_rate: str,
) -> pd.DataFrame:
    """
    Resample vehicle GPS onto the same grid as audio features.
    """
    gps_numeric = latitude_longitude_df[["Latitude", "Longitude"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    gps_numeric = gps_numeric.dropna(subset=["Latitude", "Longitude"])
    resampled = gps_numeric.resample(resample_rate).mean()
    return resampled.interpolate(method="time").dropna(subset=["Latitude", "Longitude"])


def _load_audio_power_for_node(
    base_dir: str,
    node_id: int,
    chunk_ms: int,
) -> pd.DataFrame | None:
    """
    Looks for files matching:
        YYYYMMDD_HHMMSS_*_{node_id}_respeaker.flac
    and converts them into chunked average power time series.
    """
    pattern = re.compile(rf"\d{{8}}_\d{{6}}_.*_{node_id}_respeaker\.flac$", re.IGNORECASE)
    matches = [f for f in os.listdir(base_dir) if pattern.match(f)]

    if not matches:
        print(f"No FLAC files found for node {node_id} in {base_dir}")
        return None

    node_frames = []

    for file_name in sorted(matches):
        match = re.match(
            rf"(\d{{8}})_(\d{{6}})_.*_{node_id}_respeaker\.flac$",
            file_name,
            re.IGNORECASE,
        )
        if match is None:
            continue

        date_str = match.group(1)
        time_str = match.group(2)
        start_time = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S")

        flac_audio_path = os.path.join(base_dir, file_name)
        data, samplerate = sf.read(flac_audio_path, dtype="int16")
        data = data.astype(np.float32)

        print(f"Loaded FLAC: {flac_audio_path}")

        # Match your previous behavior: if multichannel, average channels 1:5
        if data.ndim > 1:
            hi = min(data.shape[1], 5)
            lo = 1 if data.shape[1] > 1 else 0
            data = data[:, lo:hi].mean(axis=1)

        samples_per_chunk = int((chunk_ms / 1000.0) * samplerate)
        if samples_per_chunk <= 0:
            raise ValueError("samples_per_chunk must be positive.")

        num_chunks = len(data) // samples_per_chunk
        if num_chunks == 0:
            print(f"Audio too short for node {node_id} in {file_name}; skipping.")
            continue

        truncated = data[: num_chunks * samples_per_chunk]
        power = truncated**2
        reshaped = power.reshape(num_chunks, samples_per_chunk)
        averaged_power = reshaped.mean(axis=1)

        time_index = pd.date_range(
            start=start_time,
            periods=num_chunks,
            freq=f"{chunk_ms}ms",
        )
        df = pd.DataFrame(averaged_power, index=time_index, columns=[f"rpi{node_id}"])
        node_frames.append(df)

    if not node_frames:
        return None

    out = pd.concat(node_frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def plot_gap_timeline(
    df: pd.DataFrame,
    expected_freq_ms: int = 200,
    title: str = "Data Continuity Timeline",
    ax=None,
):
    """
    Visualizes time continuity / gaps for a datetime-indexed DataFrame.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index.")

    df = df.sort_index()
    if len(df) == 0:
        print("Empty DataFrame given to plot_gap_timeline.")
        return

    diffs = df.index.to_series().diff().fillna(pd.Timedelta(milliseconds=expected_freq_ms))
    segments = []
    gaps = []
    current_start = df.index[0]

    for prev_time, delta in zip(df.index[:-1], diffs.iloc[1:]):
        if delta > pd.Timedelta(milliseconds=expected_freq_ms):
            segments.append((current_start, prev_time, "continuous"))
            gap_start = prev_time
            gap_end = prev_time + delta
            segments.append((gap_start, gap_end, "gap"))
            gaps.append((gap_start, gap_end, delta))
            current_start = gap_end

    if current_start < df.index[-1]:
        segments.append((current_start, df.index[-1], "continuous"))

    top_gaps = sorted(gaps, key=lambda x: x[2], reverse=True)[:5]

    created_axis = ax is None
    if created_axis:
        _, ax = plt.subplots(figsize=(15, 2))

    for start, end, kind in segments:
        ax.plot([start, end], [1, 1], color="green" if kind == "continuous" else "red", linewidth=6)

    for start, end, delta in top_gaps:
        gap_len = int(delta.total_seconds() * 1000)
        ax.text(
            start + (end - start) / 2,
            1.02,
            f"{gap_len} ms",
            color="red",
            fontsize=9,
            ha="center",
            rotation=45,
        )

    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_xlim(df.index[0], df.index[-1])
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    continuous_patch = mpatches.Patch(color="green", label="Continuous (expected)")
    gap_patch = mpatches.Patch(color="red", label="Gap (> expected)")
    ax.legend(handles=[continuous_patch, gap_patch], loc="upper right")

    if created_axis:
        plt.tight_layout()
        plt.show()


def plot_sensor_map(gdf_nodes: gpd.GeoDataFrame, gdf_cleaned: gpd.GeoDataFrame, title: str = "Sensors and Vehicle Path"):
    """
    Plots sensor locations and the vehicle path.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    gdf_nodes.plot(ax=ax, color="red", markersize=60, label="Sensors")
    gdf_cleaned.plot(ax=ax, color="blue", markersize=3, alpha=0.5, label="Vehicle Path")

    for _, row in gdf_nodes.iterrows():
        node_id = row.get("Node #", row.get("Node", None))
        ax.annotate(str(node_id), (row.geometry.x, row.geometry.y), xytext=(5, 5), textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rssi_timeseries(
    gdf_cleaned: gpd.GeoDataFrame,
    node_list: list[int],
    max_nodes_per_fig: int = 4,
):
    """
    Plots RSSI time series in chunks of max_nodes_per_fig.
    """
    nodes = [n for n in node_list if f"rpi{n}" in gdf_cleaned.columns]
    if not nodes:
        print("No RSSI columns found to plot.")
        return

    for start in range(0, len(nodes), max_nodes_per_fig):
        chunk = nodes[start : start + max_nodes_per_fig]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(14, 3 * len(chunk)), sharex=True)
        if len(chunk) == 1:
            axes = [axes]

        for ax, node in zip(axes, chunk):
            col = f"rpi{node}"
            ax.plot(gdf_cleaned.index, gdf_cleaned[col], linewidth=0.8)
            ax.set_title(f"RSSI Time Series - Node {node}")
            ax.set_ylabel("RSSI (dB)")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()


def plot_rssi_vs_distance(
    gdf_cleaned: gpd.GeoDataFrame,
    node_list: list[int],
    ncols: int = 2,
):
    """
    Scatterplots RSSI vs distance for each node.
    """
    nodes = [
        n for n in node_list
        if f"rpi{n}" in gdf_cleaned.columns and f"distance_to_{n}" in gdf_cleaned.columns
    ]
    if not nodes:
        print("No matching RSSI/distance columns found.")
        return

    nrows = int(np.ceil(len(nodes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, node in zip(axes, nodes):
        rss_col = f"rpi{node}"
        dist_col = f"distance_to_{node}"
        sub = gdf_cleaned[[rss_col, dist_col]].dropna()

        ax.scatter(sub[dist_col], sub[rss_col], s=4, alpha=0.6)
        ax.set_xscale("log")
        ax.set_title(f"Node {node}: RSSI vs Distance")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("RSSI (dB)")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(nodes) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def load_data(
    data_dir,
    sensor_csv,
    gps_csv,
    node_list,
    chunk_ms=10,
    sample_ms=200,
    ref_distance=4.0,
    dist_tol=0.2,
    resample_rate="200ms",
    drop_nodes=None,
    plot_gaps=True,
    interpolate_gps=True,
):
    """
    Main loader.

    Returns:
        gdf_cleaned
        normalized_cleaned
        valid_indices
        gdf_nodes
    """
    drop_nodes = set(drop_nodes or [])
    available_nodes = [n for n in node_list if n not in drop_nodes]

    if len(available_nodes) == 0:
        raise ValueError("No nodes left after applying drop_nodes.")

    RESAMPLE_RATE = resample_rate
    SAMPLE_MS = sample_ms
    base_dir = data_dir

    # === Load and process audio data ===
    all_data = []
    per_node_frames = {}

    for i in available_nodes:
        df = _load_audio_power_for_node(base_dir=base_dir, node_id=i, chunk_ms=chunk_ms)
        if df is not None:
            all_data.append(df)
            per_node_frames[i] = df

    if not all_data:
        raise FileNotFoundError(f"No matching FLAC files found in {base_dir} for nodes {available_nodes}.")

    combined_df = pd.concat(all_data, axis=1).sort_index()

    for i in available_nodes:
        if i in per_node_frames:
            df = per_node_frames[i]
            print(f"Node {i}: {len(df)} rows loaded, time range: {df.index.min()} to {df.index.max()}")

    # === Load GPS Data ===
    raw_latitude_longitude_df = _read_vehicle_gps(gps_csv)
    latitude_longitude_df = (
        _resample_vehicle_gps(raw_latitude_longitude_df, RESAMPLE_RATE)
        if interpolate_gps
        else raw_latitude_longitude_df
    )

    # === Plot Data Continuity ===
    if plot_gaps:
        plot_gap_timeline(
            latitude_longitude_df,
            expected_freq_ms=SAMPLE_MS,
            title="Vehicle GPS Continuity Timeline",
        )

    # === Resample and Merge ===
    resampled_power_df = combined_df.resample(RESAMPLE_RATE).mean()
    merged_df = pd.merge(
        resampled_power_df,
        latitude_longitude_df,
        how="inner",
        left_index=True,
        right_index=True,
    )

    # === Load sensor metadata ===
    gdf_nodes = pd.read_csv(sensor_csv)
    gdf_nodes = gdf_nodes.copy()
    gdf_nodes["geometry"] = [Point(lon, lat) for lat, lon in zip(gdf_nodes["Lat"], gdf_nodes["Lon"])]
    gdf_nodes = gpd.GeoDataFrame(gdf_nodes, geometry="geometry", crs="EPSG:4326")

    # Filter nodes if requested
    if "Node #" in gdf_nodes.columns:
        gdf_nodes = gdf_nodes[gdf_nodes["Node #"].isin(available_nodes)].copy()

    # Pick local UTM automatically from mean sensor location
    mean_lat = float(gdf_nodes["Lat"].mean())
    mean_lon = float(gdf_nodes["Lon"].mean())
    local_epsg = _pick_utm_epsg(mean_lat, mean_lon)

    gdf_nodes = gdf_nodes.to_crs(epsg=local_epsg)

    gdf_vehicles = gpd.GeoDataFrame(
        merged_df.reset_index(),
        geometry=gpd.points_from_xy(merged_df["Longitude"], merged_df["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=local_epsg)

    gdf_vehicles = gdf_vehicles.rename(columns={"index": "datetime"}).set_index("datetime")
    gdf_sensors_vehicle = gdf_vehicles.drop(columns=["Latitude", "Longitude"]).copy()

    # === Compute distances to nodes ===
    for _, node in gdf_nodes.iterrows():
        node_name = node["Node #"] if "Node #" in node else node["Node"]
        gdf_vehicles[f"distance_to_{node_name}"] = gdf_vehicles.geometry.distance(node.geometry)
        gdf_sensors_vehicle[f"distance_to_{node_name}"] = gdf_vehicles[f"distance_to_{node_name}"]

    # === Convert power to dB ===
    for node in available_nodes:
        rss_col = f"rpi{node}"
        if rss_col in gdf_sensors_vehicle.columns:
            gdf_sensors_vehicle[rss_col] = 10 * np.log10(
                gdf_sensors_vehicle[rss_col].replace(0, np.nan)
            )

    # === "Normalized" output kept compatible with your previous pipeline ===
    distance_df = gdf_vehicles.drop(columns="geometry").copy()
    normalized_df = distance_df.copy()
    for node in available_nodes:
        rss_col = f"rpi{node}"
        if rss_col in distance_df.columns:
            normalized_df[rss_col] = 10 * np.log10(distance_df[rss_col].replace(0, np.nan))

    # === Deduplicate and sort ===
    gdf_cleaned = gdf_sensors_vehicle[
        ~gdf_sensors_vehicle.index.duplicated(keep="first")
    ].sort_index()

    normalized_cleaned = normalized_df[
        ~normalized_df.index.duplicated(keep="first")
    ].sort_index()

    # === Compute valid_indices ===
    # Keep your previous behavior intentionally, including the movement threshold form.
    valid_indices = []
    for i in range(1, len(gdf_cleaned)):
        t_now = gdf_cleaned.index[i]
        t_prev = gdf_cleaned.index[i - 1]

        if (t_now - t_prev) == pd.Timedelta(milliseconds=SAMPLE_MS):
            p_now = gdf_cleaned.iloc[i].geometry
            p_prev = gdf_cleaned.iloc[i - 1].geometry

            if p_now.distance(p_prev) >= 0.125 * SAMPLE_MS / 1000 * 5 / 3:
                valid_indices.append(i)

    # === Final report ===
    print("\nFinal Cleaned DataFrames Summary")
    print(f"Using nodes: {available_nodes}")
    print(f"Dropped nodes: {sorted(drop_nodes)}")
    print(f"Local projected CRS: EPSG:{local_epsg}")
    print(f"gdf_cleaned: {len(gdf_cleaned)} rows, columns: {list(gdf_cleaned.columns)}")
    print(f"normalized_cleaned: {len(normalized_cleaned)} rows, columns: {list(normalized_cleaned.columns)}")
    print(f"\nValid time indices: {len(valid_indices)} out of {len(gdf_cleaned)} total")
    print(f"First 10 valid indices: {valid_indices[:10]}")

    return gdf_cleaned, normalized_cleaned, valid_indices, gdf_nodes
