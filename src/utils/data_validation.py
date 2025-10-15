"""Data validation and integrity checks for training/evaluation datasets."""
from __future__ import annotations

import json
import os
from typing import Dict, Any, List

import numpy as np


def validate_deepstacked_samples(path: str) -> Dict[str, Any]:
    """Validate DeepStacked training/test binary files in a directory."""
    required = [
        ("train.inputs", np.float32),
        ("train.targets", np.float32),
        ("train.mask", np.float32),
        ("valid.inputs", np.float32),
        ("valid.targets", np.float32),
        ("valid.mask", np.float32),
    ]
    results = {"path": path, "files": {}, "errors": []}

    for fname, dtype in required:
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            results["errors"].append(f"Missing {fname}")
            continue
        try:
            arr = np.fromfile(fpath, dtype=dtype)
            results["files"][fname] = {"dtype": str(dtype), "size": int(arr.size)}
        except Exception as e:
            results["errors"].append(f"Error reading {fname}: {e}")

    return results


def validate_equity_table(path: str) -> Dict[str, Any]:
    """Validate equity table JSON structure."""
    if not os.path.exists(path):
        return {"path": path, "error": "missing"}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        ok = isinstance(data, dict) and len(data) > 0
        sample = dict(list(data.items())[:5]) if ok else {}
        return {"path": path, "ok": ok, "sample": sample, "count": len(data) if ok else 0}
    except Exception as e:
        return {"path": path, "error": str(e)}
