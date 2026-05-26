"""
examples/real_data_example.py
------------------------------
Load a real CSV file and run the full EEMD-ICA pipeline.

Expected CSV format (any date + price column):
    date,close
    2010-01-04,79.36
    2010-01-05,79.58
    ...

Run:
    python examples/real_data_example.py --csv your_data.csv --price-col close
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eemd_ica import EEMDICAPipeline
from eemd_ica.plotting import (
    plot_imfs, plot_cci_bar, plot_components, plot_verification_heatmap,
)
from eemd_ica.io_helpers import save_components_csv, save_results_json


def load_csv(path: str, price_col: str, date_col: str = "date"):
    """Load a CSV and return (dates, prices) as numpy arrays."""
    import csv
    dates, prices = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                prices.append(float(row[price_col]))
                dates.append(row.get(date_col, ""))
            except (ValueError, KeyError):
                continue
    return np.array(dates), np.array(prices, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="EEMD-ICA on real CSV data")
    parser.add_argument("--csv",        required=True, help="Path to price CSV file")
    parser.add_argument("--price-col",  default="close", help="Name of price column (default: close)")
    parser.add_argument("--date-col",   default="date", help="Name of date column (default: date)")
    parser.add_argument("--ensemble",   type=int, default=100, help="EEMD ensemble size")
    parser.add_argument("--threshold",  type=float, default=0.3, help="RHD threshold")
    parser.add_argument("--components", type=int, default=None, help="Number of ICs (default: auto)")
    parser.add_argument("--no-plots",   action="store_true", help="Skip matplotlib plots")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading data from: {args.csv}")
    dates, prices = load_csv(args.csv, args.price_col, args.date_col)
    print(f"Loaded {len(prices)} price observations.")

    pipeline = EEMDICAPipeline(
        ensemble_size=args.ensemble,
        noise_std=0.2,
        rhd_threshold=args.threshold,
        n_components=args.components,
        use_log_returns=True,
        standardise_input=True,
        detect_structural_break=True,
        verbose=True,
    )

    results = pipeline.fit(prices)
    print("\n" + pipeline.summary())

    # Save outputs
    out = args.output_dir
    save_components_csv(results["components"], os.path.join(out, "independent_components.csv"))
    save_results_json(
        {
            "verification": results["verification"],
            "b_coeffs": results["b_coeffs"],
            "ccis": results["ccis"],
            "elapsed": results["elapsed"],
        },
        os.path.join(out, "results_metadata.json"),
    )

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            plot_imfs(results["imfs"], original=results["preprocessed_series"]).savefig(
                os.path.join(out, "imfs.png"), dpi=150)
            plot_cci_bar(results["ccis"], threshold=pipeline.rhd_threshold).savefig(
                os.path.join(out, "cci.png"), dpi=150)
            plot_components(results["components"], original=results["preprocessed_series"]).savefig(
                os.path.join(out, "components.png"), dpi=150)
            plot_verification_heatmap(results["verification"]).savefig(
                os.path.join(out, "verification.png"), dpi=150)

            print(f"\nPlots saved to {out}/")
        except ImportError:
            print("matplotlib not installed — skipping plots.")


if __name__ == "__main__":
    main()
