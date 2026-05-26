"""
pipeline.py
-----------
EEMDICAPipeline: High-level, one-call orchestration of all five stages.

Usage
-----
>>> import numpy as np
>>> from eemd_ica import EEMDICAPipeline
>>>
>>> prices = np.loadtxt("my_prices.csv", delimiter=",", skiprows=1)
>>> pipeline = EEMDICAPipeline()
>>> results = pipeline.fit(prices)
>>> print(pipeline.summary())

The pipeline runs the following stages in order:
  I.   Period selection / structural break detection (optional)
  II.  Log-return preprocessing
  III. EEMD decomposition
  IV.  RHD filtering → VIMF generation
  V.   FastICA source separation
  VI.  Statistical verification
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any

from eemd_ica.analyzer import EEMDICAAnalyzer
from eemd_ica.preprocessing import (
    log_returns,
    standardise,
    find_structural_break,
    adf_test,
    align_series,
)


class EEMDICAPipeline:
    """
    Full EEMD-ICA factor analysis pipeline.

    Parameters
    ----------
    ensemble_size : int
        EEMD ensemble size (N). Higher = less noise, more computation.
        Default: 100.
    noise_std : float
        White noise amplitude fraction (ε). Default: 0.2.
    rhd_threshold : float
        Minimum CCI to keep an IMF as a VIMF. Default: 0.3.
    n_components : int or None
        Number of ICA components. None = auto-select. Default: None.
    use_log_returns : bool
        If True, convert price series to log returns before decomposition.
        Set False if your data is already log returns or stationary. Default: True.
    standardise_input : bool
        If True, z-score the series before EEMD. Default: True.
    detect_structural_break : bool
        If True, run Chow-test grid search and report break point. Default: False.
    random_seed : int or None
        Seed for reproducibility. Default: 42.
    verbose : bool
        Print stage progress. Default: True.

    Attributes
    ----------
    analyzer_ : EEMDICAAnalyzer
        The fitted core analyzer.
    results_  : dict
        All pipeline outputs (see get_results()).
    elapsed_  : dict
        Wall-clock time (seconds) for each stage.
    """

    def __init__(
        self,
        ensemble_size: int = 100,
        noise_std: float = 0.2,
        rhd_threshold: float = 0.3,
        n_components: Optional[int] = None,
        use_log_returns: bool = True,
        standardise_input: bool = True,
        detect_structural_break: bool = False,
        random_seed: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.ensemble_size            = ensemble_size
        self.noise_std                = noise_std
        self.rhd_threshold            = rhd_threshold
        self.n_components             = n_components
        self.use_log_returns          = use_log_returns
        self.standardise_input        = standardise_input
        self.detect_structural_break  = detect_structural_break
        self.random_seed              = random_seed
        self.verbose                  = verbose

        # Filled by fit()
        self.analyzer_: Optional[EEMDICAAnalyzer] = None
        self.results_:  Optional[Dict[str, Any]]  = None
        self.elapsed_:  Dict[str, float]           = {}

        # Preprocessing metadata
        self._mu:    Optional[float] = None
        self._sigma: Optional[float] = None
        self._preprocessed: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        time_series: np.ndarray,
        proxy_variables: Optional[np.ndarray] = None,
        proxy_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full EEMD-ICA pipeline on `time_series`.

        Parameters
        ----------
        time_series      : np.ndarray, shape (T,)
            Raw financial time series (prices or log returns).
        proxy_variables  : np.ndarray or None, shape (T', k) or (T',)
            Economic proxy variables for Stage V robust regression.
            If T' ≠ T (after preprocessing), arrays are right-aligned.
        proxy_names      : list of str or None
            Column names for proxy_variables.

        Returns
        -------
        results : dict  — same as self.results_
        """
        series = np.asarray(time_series, dtype=float).squeeze()
        if series.ndim != 1:
            raise ValueError("time_series must be 1-D.")

        # ------------------------------------------------------------------
        # Stage I: Preprocessing & structural break detection
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._log("=" * 60)
        self._log("EEMD-ICA Factor Analysis Pipeline")
        self._log("=" * 60)
        self._log(f"\n[Stage I]  Preprocessing  (T={len(series)})")

        if self.use_log_returns:
            series = log_returns(series)
            self._log(f"           → Log returns computed  (T={len(series)})")

        if self.standardise_input:
            series, self._mu, self._sigma = standardise(series)
            self._log(f"           → Standardised  (μ={self._mu:.4f}, σ={self._sigma:.4f})")

        if self.detect_structural_break:
            break_result = find_structural_break(series)
            self._log(
                f"           → Structural break: bp={break_result['best_breakpoint']} "
                f"(F={break_result['f_statistic']:.2f}, p={break_result['p_value']:.4f})"
            )
        else:
            break_result = None

        self._preprocessed = series
        self.elapsed_["preprocessing"] = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # Stage II: Align proxy variables
        # ------------------------------------------------------------------
        if proxy_variables is not None:
            proxy_variables = np.asarray(proxy_variables, dtype=float)
            if proxy_variables.ndim == 1:
                proxy_variables = proxy_variables.reshape(-1, 1)
            series, *proxy_cols = align_series(series, *[proxy_variables[:, k] for k in range(proxy_variables.shape[1])])
            proxy_variables = np.column_stack(proxy_cols) if proxy_cols else None
            self._preprocessed = series
            self._log(f"           → Proxy variables aligned  (T={len(series)})")

        # ------------------------------------------------------------------
        # Stage III: EEMD Decomposition
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._log(f"\n[Stage III] EEMD Decomposition  (ensemble_size={self.ensemble_size})")

        self.analyzer_ = EEMDICAAnalyzer(
            ensemble_size=self.ensemble_size,
            noise_std=self.noise_std,
            rhd_threshold=self.rhd_threshold,
            n_components=self.n_components,
            random_seed=self.random_seed,
        )
        imfs = self.analyzer_.decompose_eemd(series)
        self._log(f"           → {len(imfs)} IMFs extracted (incl. residual)")
        self.elapsed_["eemd"] = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # Stage IV: RHD → VIMF
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._log(f"\n[Stage IV] RHD Integration  (threshold={self.rhd_threshold})")
        vimfs = self.analyzer_.generate_vimfs()
        self._log(f"           → {len(vimfs)} VIMFs retained")
        self._log(self.analyzer_.integration_summary())
        self.elapsed_["integration"] = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # Stage V: FastICA
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._log(f"\n[Stage V]  FastICA Source Separation")
        components = self.analyzer_.extract_independent_components()
        self._log(f"           → {len(components)} ICs extracted")
        self._log(self.analyzer_.separation_summary())
        self.elapsed_["ica"] = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # Stage VI: Verification
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._log(f"\n[Stage VI] Statistical Verification")
        verification = self.analyzer_.verify(
            proxy_variables=proxy_variables,
            proxy_names=proxy_names,
        )
        self._log(self.analyzer_.verification_summary())
        self.elapsed_["verification"] = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # Collect results
        # ------------------------------------------------------------------
        self.results_ = {
            **self.analyzer_.get_results(),
            "preprocessed_series": self._preprocessed,
            "structural_break": break_result,
            "elapsed": self.elapsed_,
        }

        total = sum(self.elapsed_.values())
        self._log(f"\n{'='*60}")
        self._log(f"Pipeline complete in {total:.1f}s")
        self._log(f"{'='*60}\n")

        return self.results_

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a full text summary of the pipeline run."""
        if self.analyzer_ is None:
            return "Pipeline has not been fitted yet. Call fit() first."
        lines = [
            "=" * 60,
            "EEMD-ICA Pipeline Summary",
            "=" * 60,
            "",
            self.analyzer_.integration_summary(),
            "",
            self.analyzer_.separation_summary(),
            "",
            self.analyzer_.verification_summary(),
            "",
            "Elapsed times:",
        ]
        for stage, t in self.elapsed_.items():
            lines.append(f"  {stage:20s}: {t:6.2f}s")
        return "\n".join(lines)

    def get_results(self) -> Dict[str, Any]:
        """Return the results dict."""
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")
        return self.results_

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def __repr__(self) -> str:
        return (
            f"EEMDICAPipeline("
            f"ensemble_size={self.ensemble_size}, "
            f"noise_std={self.noise_std}, "
            f"rhd_threshold={self.rhd_threshold})"
        )
