"""
plotting.py
-----------
Visualisation utilities for EEMD-ICA results.

All functions return Matplotlib Figure objects so callers can save, display,
or further customise them.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Any
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    warnings.warn(
        "matplotlib is not installed. Plotting functions will not be available.",
        ImportWarning,
    )


def _require_matplotlib():
    if not _MPL_AVAILABLE:
        raise ImportError("Install matplotlib to use plotting utilities: pip install matplotlib")


# ---------------------------------------------------------------------------
# IMF plots
# ---------------------------------------------------------------------------

def plot_imfs(
    imfs: List[np.ndarray],
    original: Optional[np.ndarray] = None,
    title: str = "EEMD Decomposition — IMFs",
    figsize: Optional[tuple] = None,
    max_imfs: int = 12,
) -> "plt.Figure":
    """
    Plot the original series and each IMF in a stacked panel layout.

    Parameters
    ----------
    imfs     : list of np.ndarray — output of EEMDDecomposer
    original : np.ndarray or None — original time series (plotted at top)
    title    : str               — figure title
    figsize  : tuple or None     — figure size; auto-computed if None
    max_imfs : int               — cap on how many IMFs to display

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _require_matplotlib()

    n_panels = min(len(imfs), max_imfs)
    if original is not None:
        n_panels += 1

    if figsize is None:
        figsize = (14, 1.8 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    if original is not None:
        axes[ax_idx].plot(original, color="#2C3E50", linewidth=0.8)
        axes[ax_idx].set_ylabel("Original", fontsize=8)
        axes[ax_idx].set_title(title, fontsize=12, fontweight="bold")
        ax_idx += 1

    colors = plt.cm.tab10.colors
    for k, imf in enumerate(imfs[: max_imfs]):
        color = colors[k % len(colors)]
        label = "Residual" if k == len(imfs) - 1 else f"IMF-{k+1}"
        axes[ax_idx].plot(imf, color=color, linewidth=0.7)
        axes[ax_idx].set_ylabel(label, fontsize=8, rotation=0, labelpad=40, ha="right")
        axes[ax_idx].axhline(0, color="grey", linewidth=0.4, linestyle="--")
        ax_idx += 1
        if ax_idx >= n_panels:
            break

    axes[-1].set_xlabel("Time Index", fontsize=9)
    fig.tight_layout()
    return fig


def plot_cci_bar(
    ccis: np.ndarray,
    threshold: float = 0.3,
    title: str = "Contribution Coefficients (CCI)",
) -> "plt.Figure":
    """
    Bar chart of CCI values with the threshold line.

    Parameters
    ----------
    ccis      : np.ndarray — CCI for each IMF
    threshold : float      — rhd_threshold used for filtering
    title     : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _require_matplotlib()

    fig, ax = plt.subplots(figsize=(max(8, len(ccis) * 0.8), 4))

    labels = [f"IMF-{k+1}" if k < len(ccis) - 1 else "Residual" for k in range(len(ccis))]
    colors = ["#27AE60" if c >= threshold else "#E74C3C" for c in ccis]

    ax.bar(labels, ccis, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(threshold, color="#2C3E50", linewidth=1.5, linestyle="--", label=f"Threshold = {threshold}")
    ax.set_ylabel("CCI", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-1.05, 1.05)

    # Annotate bars
    for i, (lbl, val) in enumerate(zip(labels, ccis)):
        ax.text(i, val + 0.02 * np.sign(val), f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    return fig


def plot_components(
    components: np.ndarray,
    original: Optional[np.ndarray] = None,
    title: str = "Extracted Independent Components (ICs)",
    figsize: Optional[tuple] = None,
) -> "plt.Figure":
    """
    Plot each extracted IC in a stacked panel layout.

    Parameters
    ----------
    components : np.ndarray, shape (n_components, T)
    original   : np.ndarray or None
    title      : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _require_matplotlib()

    n_ic = len(components)
    n_panels = n_ic + (1 if original is not None else 0)

    if figsize is None:
        figsize = (14, 2.0 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    if original is not None:
        axes[ax_idx].plot(original, color="#2C3E50", linewidth=0.8)
        axes[ax_idx].set_ylabel("Original", fontsize=8)
        axes[ax_idx].set_title(title, fontsize=12, fontweight="bold")
        ax_idx += 1

    colors = plt.cm.Set2.colors
    for k, ic in enumerate(components):
        color = colors[k % len(colors)]
        axes[ax_idx].plot(ic, color=color, linewidth=0.8)
        axes[ax_idx].set_ylabel(f"IC-{k+1}", fontsize=9, rotation=0, labelpad=35, ha="right")
        axes[ax_idx].axhline(0, color="grey", linewidth=0.4, linestyle="--")
        ax_idx += 1

    axes[-1].set_xlabel("Time Index", fontsize=9)
    fig.tight_layout()
    return fig


def plot_verification_heatmap(
    results: List[Dict[str, Any]],
    title: str = "IC Verification Metrics",
) -> "plt.Figure":
    """
    Heatmap showing verification metrics across all ICs.

    Parameters
    ----------
    results : list of dict — output of ICVerifier.verify()
    title   : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _require_matplotlib()

    n_ic = len(results)

    metric_names = ["JB p-value", "Non-Gaussian", "Hurst", "Has Memory", "|Correlation|"]
    has_rr = any(res.get("robust_regression") for res in results)
    if has_rr:
        metric_names.append("R² (robust)")

    data = []
    for res in results:
        jb = res["jarque_bera"]
        hu = res["hurst"]
        co = res["correlation"]
        row = [
            float(jb["p_value"]),
            float(jb["is_nongaussian"]),
            float(hu["hurst"]) if not np.isnan(hu["hurst"]) else 0.0,
            float(hu["has_memory"]),
            float(co["abs_corr"]),
        ]
        if has_rr:
            rr = res.get("robust_regression") or {}
            row.append(float(rr.get("r_squared", float("nan"))) if isinstance(rr, dict) else float("nan"))
        data.append(row)

    data_arr = np.array(data, dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, len(metric_names) * 1.5), max(4, n_ic * 0.8)))
    im = ax.imshow(data_arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_ic))
    ax.set_yticklabels([f"IC-{r['ic_index']}" for r in results], fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")

    for i in range(n_ic):
        for j in range(len(metric_names)):
            val = data_arr[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if 0.2 < val < 0.8 else "white")

    fig.tight_layout()
    return fig
