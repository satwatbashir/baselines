"""Standalone HHAR loader. Adapted from Fedge-Paper/HHAR/HierFL/fedge/fedge/task.py
with all Flower / TOML / global-config dependencies removed.

Downloads HHAR from UCI on first call, parses phone+watch accelerometer +
gyroscope CSVs, resamples to a target Hz, and produces sliding-window
samples X of shape (N, 6, window_len) with integer labels y in [0..5].

Activities (ACTIVITY_ORDER): walking, sitting, standing, biking,
stairsup, stairsdown -- 6 classes.

Usage:
    X, y = load_hhar_data(data_root="./data")     # downloads + caches
    # X: (N, 6, 100) float32, y: (N,) int64
"""
from __future__ import annotations

import gc
import glob
import hashlib
import os
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset


ACTIVITY_ORDER = ["walking", "sitting", "standing", "biking", "stairsup", "stairsdown"]
NUM_CLASSES = 6
DEFAULT_SAMPLE_RATE_HZ = 50
DEFAULT_WINDOW_SECONDS = 2
DEFAULT_WINDOW_STRIDE_SECONDS = 1


def _activity_to_int(lbl: str) -> int:
    s = str(lbl).lower().strip()
    if s in ("null", "none", "", "nan"):
        return -1
    alias = {
        "walk": "walking", "cycling": "biking", "bike": "biking",
        "sit": "sitting", "stand": "standing", "upstairs": "stairsup",
        "downstairs": "stairsdown", "stairup": "stairsup", "stairdown": "stairsdown",
    }
    s = alias.get(s, s)
    if s not in ACTIVITY_ORDER:
        return -1
    return ACTIVITY_ORDER.index(s)


def _col(df: pd.DataFrame, names: List[str]) -> str:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise KeyError(f"none of {names} found in columns={list(df.columns)}")


def ensure_hhar_dataset(data_root: str, use_watches: bool = True) -> None:
    """Download + extract UCI HHAR into data_root if absent."""
    root_path = Path(data_root)
    root_path.mkdir(parents=True, exist_ok=True)
    phone_acc = list(root_path.rglob("Phones_accelerometer*.csv"))
    phone_gyro = list(root_path.rglob("Phones_gyroscope*.csv"))
    if phone_acc and phone_gyro:
        if not use_watches:
            return
        watch_acc = list(root_path.rglob("Watch_accelerometer*.csv"))
        watch_gyro = list(root_path.rglob("Watch_gyroscope*.csv"))
        if watch_acc and watch_gyro:
            return
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00344/"
    files = ["Activity recognition exp.zip"]
    if use_watches:
        files.append("Still exp.zip")
    for filename in files:
        zip_path = root_path / filename
        if not zip_path.exists():
            print(f"[hhar] downloading {filename}...")
            url = base_url + filename.replace(" ", "%20")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"[hhar] extracting {filename}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(root_path)


def _load_csvs(data_root: str, use_watches: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _load_one(pattern: str) -> pd.DataFrame:
        files = [p for p in glob.glob(os.path.join(data_root, "**", pattern), recursive=True)
                 if "Still exp" not in p]
        if not files:
            raise RuntimeError(f"Missing HHAR files for pattern: {pattern}")
        dfs = []
        for p in files:
            chunks = []
            for chunk in pd.read_csv(
                p, chunksize=50000,
                usecols=lambda c: c.strip().lower() in
                    {"user", "device", "model", "gt", "creation_time", "arrival_time", "x", "y", "z"},
                dtype={"User": "string", "Device": "string", "Model": "string", "gt": "string",
                       "x": "float32", "y": "float32", "z": "float32"},
                low_memory=False,
            ):
                chunk.columns = [c.strip() for c in chunk.columns]
                chunk["src"] = "phone" if pattern.startswith("Phones") else "watch"
                chunk = chunk.iloc[::5].copy()   # quick decimation: keep every 5th row
                chunks.append(chunk)
            if chunks:
                dfs.append(pd.concat(chunks, ignore_index=True))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    phone_acc = _load_one("Phones_accelerometer*.csv")
    phone_gyro = _load_one("Phones_gyroscope*.csv")
    if use_watches:
        watch_acc = _load_one("Watch_accelerometer*.csv")
        watch_gyro = _load_one("Watch_gyroscope*.csv")
        acc = pd.concat([phone_acc, watch_acc], ignore_index=True)
        gyro = pd.concat([phone_gyro, watch_gyro], ignore_index=True)
    else:
        acc, gyro = phone_acc, phone_gyro

    def norm(df: pd.DataFrame) -> pd.DataFrame:
        t = _col(df, ["timestamp", "Time", "time", "Creation_Time", "Arrival_Time"])
        ux = _col(df, ["User", "user"])
        dv = _col(df, ["Device", "device", "Model", "model"])
        act = _col(df, ["gt", "GT", "Activity", "activity"])
        return pd.DataFrame({
            "timestamp": pd.to_datetime(df[t], errors="coerce"),
            "user": df[ux], "device": df[dv], "activity": df[act],
            "x": df[_col(df, ["x", "X"])],
            "y": df[_col(df, ["y", "Y"])],
            "z": df[_col(df, ["z", "Z"])],
            "src": df["src"],
        }).dropna(subset=["timestamp"])

    return norm(acc), norm(gyro)


def _prep_and_resample(df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    cols = ["user", "device", "src", "timestamp", "x", "y", "z", "activity"]
    df = df[cols].dropna(subset=["timestamp"]).copy()
    df[["x", "y", "z"]] = df[["x", "y", "z"]].astype("float32", copy=False)
    df = df.sort_values(["user", "device", "src", "timestamp"])
    rule = f"{int(round(1000 / target_hz))}ms"
    out_frames = []
    for (u, d, s), g in df.groupby(["user", "device", "src"], observed=True):
        g = g.groupby("timestamp", as_index=False, observed=True).agg(
            {"x": "mean", "y": "mean", "z": "mean", "activity": "last"})
        g = g.set_index("timestamp")
        g = g[~g.index.duplicated(keep="last")].sort_index()
        num = g[["x", "y", "z"]].astype("float32").resample(rule).mean().ffill().bfill()
        act = g[["activity"]].reindex(num.index, method="ffill")
        out = num.copy()
        out["activity"] = act["activity"]
        out["user"] = u
        out["device"] = d
        out["src"] = s
        out_frames.append(out.reset_index())
    if not out_frames:
        raise RuntimeError("No resampled data produced.")
    return pd.concat(out_frames, ignore_index=True)


def _resample_merge(acc: pd.DataFrame, gyro: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    if acc.empty or gyro.empty:
        raise RuntimeError("Missing accelerometer or gyroscope CSVs.")
    a = _prep_and_resample(acc, target_hz).rename(columns={"x": "ax", "y": "ay", "z": "az"})
    g = _prep_and_resample(gyro, target_hz).rename(columns={"x": "gx", "y": "gy", "z": "gz"})
    m = pd.merge(a, g, on=["user", "device", "src", "timestamp", "activity"], how="inner").dropna()
    if m.empty:
        raise RuntimeError("Resampling produced no aligned rows.")
    m["y"] = m["activity"].map(_activity_to_int)
    m = m[(m["y"] >= 0) & (m["y"] < NUM_CLASSES)].copy()
    return m[["user", "device", "src", "timestamp", "ax", "ay", "az", "gx", "gy", "gz", "y", "activity"]]


def _windowize(df: pd.DataFrame, window_seconds: int, stride_seconds: int,
               sample_rate_hz: int) -> Tuple[np.ndarray, np.ndarray]:
    window_size = window_seconds * sample_rate_hz
    stride_size = stride_seconds * sample_rate_hz
    sensor_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
    Xs, ys = [], []
    for (user, device), group in df.groupby(["user", "device"]):
        if len(group) < window_size:
            continue
        sensor = group[sensor_cols].values.astype(np.float32)
        acts = group["activity"].values
        for i in range(0, len(sensor) - window_size + 1, stride_size):
            window = sensor[i:i + window_size]
            lbls = [_activity_to_int(l) for l in acts[i:i + window_size]]
            valid = [l for l in lbls if 0 <= l < NUM_CLASSES]
            if not valid:
                continue
            maj = int(np.bincount(np.array(valid, dtype=np.int64)).argmax())
            Xs.append(window.T)   # (6, window_size)
            ys.append(maj)
    if not Xs:
        raise RuntimeError("No windows could be created.")
    return np.stack(Xs, axis=0), np.array(ys, dtype=np.int64)


def load_hhar_data(
    data_root: str = "./data",
    use_watches: bool = True,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    window_stride_seconds: int = DEFAULT_WINDOW_STRIDE_SECONDS,
    cache_dir: str = "./data/hhar_cache",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and process HHAR data. Caches the windowed arrays under cache_dir.

    Returns:
        X: (N, 6, window_seconds * sample_rate_hz) float32
        y: (N,) int64 in [0, 5]
    """
    key = hashlib.md5(
        f"{data_root}_{use_watches}_{sample_rate_hz}_{window_seconds}_{window_stride_seconds}".encode()
    ).hexdigest()[:8]
    cache_path = Path(cache_dir) / f"hhar_{key}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"[hhar] loading cached windows from {cache_path}")
        data = np.load(cache_path, mmap_mode="r")
        return np.asarray(data["X"]), np.asarray(data["y"])
    ensure_hhar_dataset(data_root, use_watches=use_watches)
    acc, gyro = _load_csvs(data_root, use_watches=use_watches)
    merged = _resample_merge(acc, gyro, sample_rate_hz)
    X, y = _windowize(merged, window_seconds, window_stride_seconds, sample_rate_hz)
    del merged
    gc.collect()
    print(f"[hhar] caching {X.shape} windows to {cache_path}")
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


class HHARTensorDataset(Dataset):
    """Wraps (X, y) numpy arrays in a torch Dataset with per-channel z-score
    normalization (stats computed once on the train half passed in)."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 means: np.ndarray | None = None, stds: np.ndarray | None = None,
                 compute_stats: bool = True):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        if means is not None and stds is not None:
            self.means = means.reshape(-1, 1).astype(np.float32)
            self.stds = stds.reshape(-1, 1).astype(np.float32)
        elif compute_stats:
            n = min(20000, len(self.X))
            rng = np.random.default_rng(0)
            idx = rng.choice(len(self.X), n, replace=False)
            sub = self.X[idx]
            self.means = sub.mean(axis=(0, 2)).reshape(-1, 1).astype(np.float32)
            stds = sub.std(axis=(0, 2))
            stds = np.where(stds == 0, 1.0, stds).astype(np.float32)
            self.stds = stds.reshape(-1, 1)
        else:
            self.means = np.zeros((self.X.shape[1], 1), dtype=np.float32)
            self.stds = np.ones((self.X.shape[1], 1), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (self.X[idx] - self.means) / self.stds
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)
