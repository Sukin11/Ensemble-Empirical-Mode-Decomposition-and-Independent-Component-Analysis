"""
integration.py
--------------
Integration Layer: RHD-based IMF contribution scoring and VIMF generation.

After EEMD yields a set of IMFs {c_1, ..., c_N, r}, this layer:
  1. Computes the Contribution Coefficient (CCI) for each IMF via the
     Relative Hamming Distance (RHD) method.
  2. Compares each CCI against a hard threshold (rhd_threshold).
  3. Merges low-CCI IMFs into a single "noise" component.
  4. Returns the remaining meaningful IMFs as Variational IMFs (VIMFs).

Relative Hamming Distance (RHD):
---------------------------------
Given signal x(t) and IMF c(t), define the binary direction-change function:

    R_x(t) = 1  if x(t) - x(t-1) > 0  else  0
    R_c(t) = 1  if c(t) - c(t-1) > 0  else  0

The RHD (or Hamming distance normalised to [0,1]) between x and c is:

    RHD(x, c) = (1/T) * sum_{t=2}^{T} |R_x(t) - R_c(t)|

The Contribution Coefficient (CCI) is then:

    CCI(c) = 1 - 2 * RHD(x, c)

CCI ∈ [-1, 1].  A value close to +1 means the IMF co-moves strongly with
the original signal (meaningful).  Values near 0 or negative indicate noise.
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


class RHDIntegrator:
    """
    RHD-based IMF filter and VIMF generator.

    Parameters
    ----------
    rhd_threshold : float
        Minimum CCI for an IMF to be kept as a VIMF.
        IMFs with CCI < rhd_threshold are merged into a noise component.
        Typical values: 0.2 (aggressive filtering) or 0.3 (conservative).
        Use 0.0 to keep all IMFs (no filtering).
    include_residual : bool
        Whether to always retain the trend residual regardless of its CCI.
        Default: True (residual often carries the long-run trend factor).

    Attributes
    ----------
    ccis_ : np.ndarray
        Contribution coefficients for each IMF (after fitting).
    vimfs_ : list of np.ndarray
        Filtered, meaningful IMFs.
    noise_component_ : np.ndarray or None
        Sum of all low-CCI IMFs (the merged "noise" series).
        None if no IMFs fell below the threshold.
    kept_indices_ : list of int
        Original IMF indices that passed the threshold.
    merged_indices_ : list of int
        Original IMF indices that were merged into noise_component_.
    """

    def __init__(self, rhd_threshold: float = 0.3, include_residual: bool = True):
        if not (-1.0 <= rhd_threshold <= 1.0):
            raise ValueError("rhd_threshold must be in [-1, 1].")
        self.rhd_threshold = rhd_threshold
        self.include_residual = include_residual

        # Fitted attributes
        self.ccis_: Optional[np.ndarray] = None
        self.vimfs_: Optional[List[np.ndarray]] = None
        self.noise_component_: Optional[np.ndarray] = None
        self.kept_indices_: Optional[List[int]] = None
        self.merged_indices_: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        imfs: List[np.ndarray],
        original: np.ndarray,
    ) -> "RHDIntegrator":
        """
        Compute CCIs and partition IMFs into VIMFs and noise.

        Parameters
        ----------
        imfs     : list of np.ndarray  — output of EEMDDecomposer (includes residual)
        original : np.ndarray          — the original (un-noised) time series

        Returns
        -------
        self
        """
        original = np.asarray(original, dtype=float)
        imfs = [np.asarray(c, dtype=float) for c in imfs]

        if len(imfs) == 0:
            raise ValueError("imfs list is empty.")

        # The last element from EEMD is the residual (trend)
        n_imfs = len(imfs)
        residual_idx = n_imfs - 1

        # Compute CCI for every IMF
        self.ccis_ = np.array(
            [self._calculate_cci(imfs[k], original) for k in range(n_imfs)]
        )

        kept: List[int] = []
        merged: List[int] = []

        for k in range(n_imfs):
            # Always keep the residual if requested
            if k == residual_idx and self.include_residual:
                kept.append(k)
                continue
            if self.ccis_[k] >= self.rhd_threshold:
                kept.append(k)
            else:
                merged.append(k)

        self.kept_indices_ = kept
        self.merged_indices_ = merged

        # Build VIMFs
        self.vimfs_ = [imfs[k] for k in kept]

        # Build noise component
        if merged:
            self.noise_component_ = np.sum([imfs[k] for k in merged], axis=0)
        else:
            self.noise_component_ = None

        if len(self.vimfs_) == 0:
            warnings.warn(
                "All IMFs were merged into noise. Consider lowering rhd_threshold.",
                UserWarning,
                stacklevel=2,
            )

        return self

    def fit_transform(
        self,
        imfs: List[np.ndarray],
        original: np.ndarray,
    ) -> List[np.ndarray]:
        """Fit and return the list of VIMFs."""
        return self.fit(imfs, original).vimfs_

    def summary(self) -> str:
        """Human-readable summary of the CCI results."""
        self._check_fitted()
        lines = ["RHD Integration Summary", "=" * 40]
        for k, cci in enumerate(self.ccis_):
            tag = "KEPT" if k in self.kept_indices_ else "MERGED"
            label = "residual" if k == len(self.ccis_) - 1 else f"IMF-{k+1}"
            lines.append(f"  {label:12s}  CCI={cci:+.4f}  [{tag}]")
        lines.append(f"\n  Threshold : {self.rhd_threshold}")
        lines.append(f"  VIMFs kept: {len(self.vimfs_)}")
        lines.append(f"  Merged    : {len(self.merged_indices_)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_rhd(imf: np.ndarray, original: np.ndarray) -> float:
        """
        Compute the Relative Hamming Distance between an IMF and the original series.

        Parameters
        ----------
        imf      : 1-D array — an individual IMF
        original : 1-D array — the original time series

        Returns
        -------
        rhd : float ∈ [0, 1]
        """
        imf = np.asarray(imf, dtype=float)
        original = np.asarray(original, dtype=float)

        if len(imf) != len(original):
            raise ValueError("imf and original must have the same length.")
        if len(imf) < 2:
            raise ValueError("Signal must have at least 2 points.")

        # Direction-change binary vectors (differences)
        R_orig = (np.diff(original) > 0).astype(int)
        R_imf  = (np.diff(imf)      > 0).astype(int)

        rhd = np.mean(np.abs(R_orig - R_imf))
        return float(rhd)

    @classmethod
    def _calculate_cci(cls, imf: np.ndarray, original: np.ndarray) -> float:
        """CCI = 1 - 2 * RHD."""
        return 1.0 - 2.0 * cls.calculate_rhd(imf, original)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.ccis_ is None:
            raise RuntimeError("RHDIntegrator has not been fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        return f"RHDIntegrator(rhd_threshold={self.rhd_threshold}, include_residual={self.include_residual})"
