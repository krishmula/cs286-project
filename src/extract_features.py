"""Extract handcrafted time-domain and frequency-domain features from cached windows.

Computes an 84-dimensional feature vector for every window:
  - 12 channels × 4 time-domain features (mean, std, RMS, ZCR)
  - 12 channels × 3 frequency-domain features (dominant frequency, spectral energy,
    spectral entropy)

Channel order (matches x_fusion layout):
  0-5  : phone  (ax, ay, az, gx, gy, gz)
  6-11 : watch  (ax, ay, az, gx, gy, gz)

Outputs one .npz file per split to artifacts/classical/:
  X             : (n_samples, 84)  float32
  y             : (n_samples,)     int64   label indices
  subjects      : (n_samples,)     int64   subject IDs
  feature_names : (84,)            object  readable names

Usage:
    python -m src.extract_features [--project-root PATH] [--output-dir PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import WindowRepository
from .runtime import resolve_project_paths

# ── channel metadata ──────────────────────────────────────────────────────────

CHANNEL_NAMES = [
    "phone_ax", "phone_ay", "phone_az",
    "phone_gx", "phone_gy", "phone_gz",
    "watch_ax", "watch_ay", "watch_az",
    "watch_gx", "watch_gy", "watch_gz",
]

TIME_FEATURES = ["mean", "std", "rms", "zcr"]
FREQ_FEATURES = ["dom_freq", "spectral_energy", "spectral_entropy"]
ALL_SUFFIXES  = TIME_FEATURES + FREQ_FEATURES  # 7 per channel

TARGET_HZ     = 20
WINDOW_LEN    = 60  # samples per window at 20 Hz


def _feature_names() -> list[str]:
    return [f"{ch}_{feat}" for ch in CHANNEL_NAMES for feat in ALL_SUFFIXES]


# ── per-channel feature functions ─────────────────────────────────────────────

def _mean(ch: np.ndarray) -> float:
    return float(np.mean(ch))


def _std(ch: np.ndarray) -> float:
    return float(np.std(ch))


def _rms(ch: np.ndarray) -> float:
    return float(np.sqrt(np.mean(ch ** 2)))


def _zcr(ch: np.ndarray) -> float:
    # fraction of consecutive sample pairs that cross zero
    if len(ch) < 2:
        return 0.0
    return float(np.sum(np.diff(np.sign(ch)) != 0) / (len(ch) - 1))


def _dom_freq(ch: np.ndarray) -> float:
    freqs = np.fft.rfftfreq(len(ch), d=1.0 / TARGET_HZ)
    mags  = np.abs(np.fft.rfft(ch))
    # exclude DC bin (index 0)
    if len(mags) <= 1:
        return 0.0
    peak_bin = int(np.argmax(mags[1:])) + 1
    return float(freqs[peak_bin])


def _spectral_energy(ch: np.ndarray) -> float:
    mags = np.abs(np.fft.rfft(ch))
    return float(np.sum(mags ** 2) / len(ch))


def _spectral_entropy(ch: np.ndarray) -> float:
    mags   = np.abs(np.fft.rfft(ch))
    power  = mags ** 2
    total  = power.sum()
    if total < 1e-12:
        return 0.0
    p = power / total
    return float(-np.sum(p * np.log2(p + 1e-12)))


def _channel_features(ch: np.ndarray) -> list[float]:
    """Return the 7-element feature vector for a single channel (60 samples)."""
    return [
        _mean(ch),
        _std(ch),
        _rms(ch),
        _zcr(ch),
        _dom_freq(ch),
        _spectral_energy(ch),
        _spectral_entropy(ch),
    ]


# ── window-level feature extraction ──────────────────────────────────────────

def extract_window_features(x: np.ndarray) -> np.ndarray:
    """Extract 84-dim feature vector from a (12, 60) window array."""
    assert x.shape == (12, WINDOW_LEN), f"Expected (12, {WINDOW_LEN}), got {x.shape}"
    feats: list[float] = []
    for ch_idx in range(12):
        feats.extend(_channel_features(x[ch_idx]))
    return np.array(feats, dtype=np.float32)


def extract_split_features(
    split_data,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for all samples in a split.

    Returns
    -------
    X        : (n, 84) float32
    y        : (n,)    int64
    subjects : (n,)    int64
    """
    n = len(split_data.samples)
    X = np.empty((n, len(feature_names)), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    subjects = np.empty(n, dtype=np.int64)

    for i, sample in enumerate(split_data.samples):
        X[i] = extract_window_features(sample.x_fusion)
        y[i] = sample.label_idx
        subjects[i] = sample.subject_id

    return X, y, subjects


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir",   type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args  = parse_args()
    paths = resolve_project_paths(args.project_root)
    out_dir = (args.output_dir or paths.artifacts_root / "classical").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = WindowRepository(paths.windows_root)
    names = _feature_names()
    names_arr = np.array(names, dtype=object)

    print(f"Feature names ({len(names)}): {names[:4]} ... {names[-2:]}")

    for split in ["train", "val", "test"]:
        print(f"[extract_features] loading split '{split}' ...")
        split_data = repo.load_split(split)
        print(f"  {len(split_data.samples)} windows")

        X, y, subjects = extract_split_features(split_data, names)
        out_path = out_dir / f"features_{split}.npz"
        np.savez(out_path, X=X, y=y, subjects=subjects, feature_names=names_arr)
        print(f"  saved → {out_path}  X.shape={X.shape}")

    print("[extract_features] done.")


if __name__ == "__main__":
    main()
