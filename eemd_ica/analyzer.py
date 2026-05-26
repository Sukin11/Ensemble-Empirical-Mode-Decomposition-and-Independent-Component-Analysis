"""
analyzer.py
-----------
EEMDICAAnalyzer: The core execution engine.

This class mirrors the API described in the Technical Architecture Document
and exposes each stage of the pipeline as a discrete method, giving users
fine-grained control over the decomposition workflow.

For a fully automated run, prefer EEMDICAPipeline (pipeline.py).
"""

import numpy as np
from typing import List, Optional, Dict, Any

from eemd_ica.eemd import EEMDDecomposer
from eemd_ica.rhd import RHDIntegrator
from eemd_ica.ica import ICASourceSeparator
from eemd_ica.metrics import ICVerifier


class EEMDICAAnalyzer:
    """
    Core EEMD-ICA execution engine.

    This class exposes each pipeline stage as an individual method, allowing
    users to inspect intermediate results, swap components, and experiment
    with parameters at every step.

    Parameters
    ----------
    ensemble_size : int
        Number of EEMD ensemble trials (N). Default: 100.
    noise_std : float
        White noise amplitude as fraction of signal std (ε). Default: 0.2.
    rhd_threshold : float
        CCI threshold for VIMF selection. Default: 0.3.
    n_components : int or None
        Number of ICA components. None = auto-select. Default: None.
    random_seed : int or None
        Seed for reproducibility. Default: 42.

    Attributes
    ----------
    imfs_        : list of np.ndarray — EEMD output
    ccis_        : np.ndarray         — Contribution coefficients
    vimfs_       : list of np.ndarray — Filtered VIMFs
    components_  : np.ndarray         — Extracted ICs (n_components × T)
    b_coeffs_    : np.ndarray         — Transformation coefficients
    verification_: list of dict       — Verification metric results
    """

    def __init__(
        self,
        ensemble_size: int = 100,
        noise_std: float = 0.2,
        rhd_threshold: float = 0.3,
        n_components: Optional[int] = None,
        random_seed: Optional[int] = 42,
    ):
        self.ensemble_size  = ensemble_size
        self.noise_std      = noise_std
        self.rhd_threshold  = rhd_threshold
        self.n_components   = n_components
        self.random_seed    = random_seed

        # Sub-module instances (created lazily)
        self._eemd: Optional[EEMDDecomposer]       = None
        self._rhd:  Optional[RHDIntegrator]         = None
        self._ica:  Optional[ICASourceSeparator]    = None
        self._verifier: Optional[ICVerifier]        = None

        # Result stores
        self.imfs_:         Optional[List[np.ndarray]] = None
        self.ccis_:         Optional[np.ndarray]       = None
        self.vimfs_:        Optional[List[np.ndarray]] = None
        self.components_:   Optional[np.ndarray]       = None
        self.b_coeffs_:     Optional[np.ndarray]       = None
        self.verification_: Optional[List[Dict]]       = None

        self._original: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Stage I — EEMD Decomposition
    # ------------------------------------------------------------------

    def decompose_eemd(self, time_series: np.ndarray) -> List[np.ndarray]:
        """
        Stage I: Decompose `time_series` using EEMD.

        Parameters
        ----------
        time_series : np.ndarray, shape (T,)

        Returns
        -------
        imfs : list of np.ndarray — IMFs (last element = trend residual)
        """
        self._original = np.asarray(time_series, dtype=float)
        self._eemd = EEMDDecomposer(
            ensemble_size=self.ensemble_size,
            noise_std=self.noise_std,
            random_seed=self.random_seed,
        )
        self.imfs_ = self._eemd.fit_transform(self._original)
        return self.imfs_

    # ------------------------------------------------------------------
    # Stage II — RHD Contribution Coefficient
    # ------------------------------------------------------------------

    def calculate_rhd_contribution(
        self,
        imf: np.ndarray,
        original: np.ndarray,
    ) -> float:
        """
        Compute the Contribution Coefficient (CCI) for a single IMF.

        This is a utility method. For batch processing of all IMFs,
        call generate_vimfs() which handles the full set.

        Parameters
        ----------
        imf      : np.ndarray — a single IMF
        original : np.ndarray — the original time series

        Returns
        -------
        cci : float ∈ [-1, 1]
        """
        return RHDIntegrator._calculate_cci(
            np.asarray(imf, dtype=float),
            np.asarray(original, dtype=float),
        )

    # ------------------------------------------------------------------
    # Stage III — VIMF Generation
    # ------------------------------------------------------------------

    def generate_vimfs(self) -> List[np.ndarray]:
        """
        Stage III: Filter IMFs by CCI and produce VIMFs.

        Must be called after decompose_eemd().

        Returns
        -------
        vimfs : list of np.ndarray
        """
        if self.imfs_ is None or self._original is None:
            raise RuntimeError("Call decompose_eemd() before generate_vimfs().")

        self._rhd = RHDIntegrator(rhd_threshold=self.rhd_threshold)
        self._rhd.fit(self.imfs_, self._original)

        self.ccis_  = self._rhd.ccis_
        self.vimfs_ = self._rhd.vimfs_

        return self.vimfs_

    # ------------------------------------------------------------------
    # Stage IV — ICA Source Separation
    # ------------------------------------------------------------------

    def extract_independent_components(self) -> np.ndarray:
        """
        Stage IV: Apply FastICA to VIMFs to extract ICs.

        Must be called after generate_vimfs().

        Returns
        -------
        components : np.ndarray, shape (n_components, T)
        """
        if self.vimfs_ is None:
            raise RuntimeError("Call generate_vimfs() before extract_independent_components().")

        self._ica = ICASourceSeparator(
            n_components=self.n_components,
            random_state=self.random_seed,
        )
        self.components_ = self._ica.fit_transform(self.vimfs_)
        self.b_coeffs_   = self._ica.transformation_coefficients_

        return self.components_

    # ------------------------------------------------------------------
    # Stage V — Verification
    # ------------------------------------------------------------------

    def verify(
        self,
        proxy_variables: Optional[np.ndarray] = None,
        proxy_names: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Stage V: Run statistical verification on extracted ICs.

        Parameters
        ----------
        proxy_variables : np.ndarray or None
            Economic proxy data, shape (T, k) or (T,).
        proxy_names     : list of str or None
            Names of the proxy columns.

        Returns
        -------
        verification_results : list of dict
        """
        if self.components_ is None or self._original is None:
            raise RuntimeError("Call extract_independent_components() before verify().")

        self._verifier = ICVerifier(
            original=self._original,
            proxy_variables=proxy_variables,
            proxy_names=proxy_names,
        )
        self.verification_ = self._verifier.verify(self.components_)
        return self.verification_

    # ------------------------------------------------------------------
    # Convenience summaries
    # ------------------------------------------------------------------

    def integration_summary(self) -> str:
        """Print the RHD integration summary."""
        if self._rhd is None:
            raise RuntimeError("Call generate_vimfs() first.")
        return self._rhd.summary()

    def separation_summary(self) -> str:
        """Print the ICA separation summary."""
        if self._ica is None:
            raise RuntimeError("Call extract_independent_components() first.")
        return self._ica.summary()

    def verification_summary(self) -> str:
        """Print the verification metrics table."""
        if self._verifier is None:
            raise RuntimeError("Call verify() first.")
        return self._verifier.summary()

    def get_results(self) -> Dict[str, Any]:
        """
        Collect all results into a single dict for export.

        Returns
        -------
        dict with keys:
            'imfs', 'ccis', 'vimfs', 'components', 'b_coeffs', 'verification'
        """
        return {
            "imfs":         self.imfs_,
            "ccis":         self.ccis_,
            "vimfs":        self.vimfs_,
            "components":   self.components_,
            "b_coeffs":     self.b_coeffs_,
            "verification": self.verification_,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EEMDICAAnalyzer("
            f"ensemble_size={self.ensemble_size}, "
            f"noise_std={self.noise_std}, "
            f"rhd_threshold={self.rhd_threshold})"
        )
