"""
io_helpers.py
-------------
Save and load pipeline results to/from disk.

Supported formats:
  - NumPy .npz  (fast, compact, default)
  - JSON        (human-readable metadata)
  - CSV         (per-IC time series, easy to import in Excel / R)
"""

import json
import os
import numpy as np
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results_npz(results: Dict[str, Any], path: str) -> None:
    """
    Save pipeline results to a compressed NumPy archive (.npz).

    Parameters
    ----------
    results : dict — typically EEMDICAPipeline.results_
    path    : str  — file path (extension .npz added if absent)
    """
    if not path.endswith(".npz"):
        path += ".npz"

    arrays = {}
    for key, val in results.items():
        if isinstance(val, np.ndarray):
            arrays[key] = val
        elif isinstance(val, list) and all(isinstance(v, np.ndarray) for v in val):
            for i, arr in enumerate(val):
                arrays[f"{key}_{i}"] = arr
        # Non-array items (dicts, lists of dicts) are skipped — save separately as JSON

    np.savez_compressed(path, **arrays)
    print(f"[io] NumPy arrays saved to {path}")


def save_results_json(results: Dict[str, Any], path: str) -> None:
    """
    Save non-array pipeline metadata (verification results, CCIs, etc.) to JSON.

    Parameters
    ----------
    results : dict
    path    : str — file path (extension .json added if absent)
    """
    if not path.endswith(".json"):
        path += ".json"

    serialisable = _to_serialisable(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[io] Metadata saved to {path}")


def save_components_csv(
    components: np.ndarray,
    path: str,
    index: Optional[np.ndarray] = None,
) -> None:
    """
    Save IC matrix to CSV (rows = time, columns = IC-1, IC-2, ...).

    Parameters
    ----------
    components : np.ndarray, shape (n_components, T)
    path       : str
    index      : optional 1-D array for the time index column
    """
    import csv

    n_components, T = components.shape
    if not path.endswith(".csv"):
        path += ".csv"

    header = (["date"] if index is not None else []) + [f"IC_{k+1}" for k in range(n_components)]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(T):
            row = ([index[t]] if index is not None else []) + [components[k, t] for k in range(n_components)]
            writer.writerow(row)

    print(f"[io] Components CSV saved to {path}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results_npz(path: str) -> Dict[str, np.ndarray]:
    """Load a .npz archive into a dict of arrays."""
    data = np.load(path, allow_pickle=False)
    return dict(data)


def load_results_json(path: str) -> Dict[str, Any]:
    """Load a JSON metadata file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
