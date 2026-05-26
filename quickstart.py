"""
examples/quickstart.py
-----------------------
Quickstart: run EEMD-ICA on synthetic oil price data.

Run:
    python examples/quickstart.py
"""

import numpy as np
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eemd_ica.pipeline import EEMDICAPipeline
from eemd_ica.plotting import (
    plot_imfs,
    plot_cci_bar,
    plot_components,
    plot_verification_heatmap,
)


# ---------------------------------------------------------------------------
# 1. Generate synthetic oil price data
# ---------------------------------------------------------------------------
np.random.seed(0)
T = 500

t = np.linspace(0, 4 * np.pi, T)

# Simulate: trend + business cycle + short-cycle + noise
trend      = 0.05 * t + 50
cycle_long = 8 * np.sin(t)
cycle_short= 3 * np.sin(4 * t + 0.5)
noise      = np.random.normal(0, 1.5, T)

prices = trend + cycle_long + cycle_short + noise
prices = np.abs(prices)  # keep prices positive

print(f"Synthetic price series: T={T}, range=[{prices.min():.2f}, {prices.max():.2f}]")


# ---------------------------------------------------------------------------
# 2. (Optional) synthetic GDP proxy
# ---------------------------------------------------------------------------
gdp_growth = 0.003 * t + np.random.normal(0, 0.01, T)  # noisy upward trend


# ---------------------------------------------------------------------------
# 3. Run the pipeline
# ---------------------------------------------------------------------------
pipeline = EEMDICAPipeline(
    ensemble_size=50,          # reduce for faster demo; use 100+ in practice
    noise_std=0.2,
    rhd_threshold=0.3,
    n_components=None,         # auto-select
    use_log_returns=True,
    standardise_input=True,
    detect_structural_break=True,
    verbose=True,
)

results = pipeline.fit(
    time_series=prices,
    proxy_variables=gdp_growth,
    proxy_names=["GDP_growth"],
)


# ---------------------------------------------------------------------------
# 4. Plot results
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    # IMF decomposition
    fig1 = plot_imfs(results["imfs"], original=results["preprocessed_series"])
    fig1.savefig("imfs.png", dpi=150)
    print("\nSaved: imfs.png")

    # CCI bar chart
    fig2 = plot_cci_bar(results["ccis"], threshold=pipeline.rhd_threshold)
    fig2.savefig("cci.png", dpi=150)
    print("Saved: cci.png")

    # Independent components
    fig3 = plot_components(results["components"], original=results["preprocessed_series"])
    fig3.savefig("components.png", dpi=150)
    print("Saved: components.png")

    # Verification heatmap
    fig4 = plot_verification_heatmap(results["verification"])
    fig4.savefig("verification.png", dpi=150)
    print("Saved: verification.png")

    plt.show()

except ImportError:
    print("\nmatplotlib not installed — skipping plots.")


# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------
from eemd_ica.io_helpers import save_results_json, save_components_csv

save_results_json(
    {
        "verification": results["verification"],
        "b_coeffs": results["b_coeffs"],
        "ccis": results["ccis"],
        "elapsed": results["elapsed"],
    },
    "results_metadata.json",
)

save_components_csv(results["components"], "independent_components.csv")

print("\nDone. Check results_metadata.json and independent_components.csv")
