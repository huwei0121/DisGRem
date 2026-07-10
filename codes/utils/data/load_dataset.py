"""
load_dataset.py - Load and preprocess standard ML datasets (LibSVM format).
Ported from MATLAB: load_dataset.m

Supports downloading from LIBSVM repository and caching locally.
"""

from __future__ import annotations
import os
import ssl
import urllib.request
from functools import lru_cache
import numpy as np
from scipy.sparse import issparse
from pathlib import Path

_SSL_CTX = ssl.create_default_context()


_LIBSVM_DATASETS = {
    "a9a", "w8a", "ijcnn1", "mushrooms", "a4a", "svmguide3", "w5a"
}

_BASE_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"


@lru_cache(maxsize=16)
def load_dataset(name: str,
                 standardize: str = "none",
                 label_style: str = "raw",
                 data_dir: str | None = None) -> tuple:
    """
    Load a dataset by name.

    Parameters
    ----------
    name         : dataset name, e.g. 'a9a', or path to local .csv/.libsvm
    standardize  : 'none' | 'zscore' | 'maxabs'
    label_style  : 'raw' | 'pm1' | '01'
    data_dir     : local cache directory

    Returns
    -------
    A : (m, d) float64 ndarray
    b : (m,) float64 ndarray
    """
    if data_dir is None:
        _pkg_root = Path(__file__).resolve().parent.parent.parent
        data_dir = str(_pkg_root / "data")
    os.makedirs(data_dir, exist_ok=True)
    name_l = name.lower()

    # ── resolve file path ────────────────────────────────────────────────
    if name_l in _LIBSVM_DATASETS:
        local_path = os.path.join(data_dir, f"{name_l}.libsvm")
        if not os.path.exists(local_path):
            url = _BASE_URL + name_l
            print(f"Downloading {name_l} from {url} ...")
            try:
                req = urllib.request.urlopen(url, context=_SSL_CTX, timeout=30)
                with open(local_path, "wb") as f:
                    f.write(req.read())
            except Exception as e:
                raise RuntimeError(f"Failed to download {name_l}: {e}")
        file_type = "libsvm"
        file_path = local_path
    elif name.endswith(".libsvm") or name.endswith(".svm"):
        file_type = "libsvm"
        file_path = name if os.path.exists(name) else os.path.join(data_dir, name)
    elif name.endswith(".csv") or name.endswith(".txt"):
        file_type = "csv"
        file_path = name if os.path.exists(name) else os.path.join(data_dir, name)
    else:
        raise ValueError(f"Cannot determine file type for '{name}'")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    # ── load ─────────────────────────────────────────────────────────────
    if file_type == "libsvm":
        b_raw, A_sparse = _libsvm_read(file_path)
        A = A_sparse.toarray() if issparse(A_sparse) else np.array(A_sparse)
        b = np.array(b_raw, dtype=float)
    else:
        T = np.loadtxt(file_path, delimiter=",")
        b = T[:, -1].astype(float)
        A = T[:, :-1].astype(float)

    m, d = A.shape

    # ── standardise ──────────────────────────────────────────────────────
    if standardize.lower() == "zscore":
        mu = A.mean(axis=0)
        sigma = A.std(axis=0) + 1e-15
        A = (A - mu) / sigma
    elif standardize.lower() == "maxabs":
        scale = np.abs(A).max(axis=0) + 1e-15
        A = A / scale

    # ── label mapping ────────────────────────────────────────────────────
    uniq = np.unique(b)
    if label_style.lower() == "pm1":
        if set(uniq).issubset({0, 1}):
            b = 2 * b - 1
    elif label_style.lower() == "01":
        if set(uniq).issubset({-1, 1}):
            b = (b + 1) / 2

    A.setflags(write=False)
    b.setflags(write=False)
    print(f"Loaded {name_l}: m={m}, d={d}")
    return A, b


# ─── pure-python LibSVM reader ────────────────────────────────────────────────
def _libsvm_read(fname: str):
    """Parse a LibSVM sparse text file → (labels, scipy.sparse.csr_matrix)."""
    from scipy.sparse import csr_matrix

    rows, cols, vals, labels = [], [], [], []
    row = 0
    with open(fname, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            labels.append(float(parts[0]))
            for tok in parts[1:]:
                if ":" not in tok:
                    continue
                idx_s, val_s = tok.split(":", 1)
                col = int(idx_s) - 1       # 0-indexed
                val = float(val_s)
                rows.append(row)
                cols.append(col)
                vals.append(val)
            row += 1

    if not cols:
        X = csr_matrix((row, 1))
    else:
        n_col = max(cols) + 1
        X = csr_matrix((vals, (rows, cols)), shape=(row, n_col))

    return np.array(labels), X
