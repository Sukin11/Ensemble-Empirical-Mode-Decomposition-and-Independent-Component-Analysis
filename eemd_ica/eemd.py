"""
decomposition.py
----------------
Decomposition Layer: Ensemble Empirical Mode Decomposition (EEMD).

EEMD resolves the mode-mixing problem of standard EMD by adding white noise
trials and averaging the resulting IMFs across all ensemble members.

Algorithm (Wu & Huang, 2009):
    1. Add white noise n_i(t) with amplitude epsilon to x(t)
    2. Decompose x_i(t) = x(t) + n_i(t) using standard EMD
    3. Repeat N times with different noise realizations
    4. Average the k-th IMFs across all N trials: c_k(t) = (1/N) * sum(c_k^i(t))
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


# ---------------------------------------------------------------------------
# Pure EMD (sifting algorithm) — used internally by EEMD
# ---------------------------------------------------------------------------

def _find_extrema(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Locate local maxima and minima of a 1-D signal.

    Returns
    -------
    max_idx, max_val, min_idx, min_val : np.ndarray
    """
    # Maxima
    max_idx = np.where(
        (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
    )[0] + 1
    # Minima
    min_idx = np.where(
        (signal[1:-1] < signal[:-2]) & (signal[1:-1] < signal[2:])
    )[0] + 1

    return max_idx, signal[max_idx], min_idx, signal[min_idx]


def _interpolate_envelope(
    idx: np.ndarray,
    val: np.ndarray,
    length: int,
    kind: str = "cubic",
) -> np.ndarray:
    """
    Interpolate sparse extrema into a continuous envelope using cubic splines.

    Parameters
    ----------
    idx   : indices of extrema
    val   : values at those indices
    length: total signal length
    kind  : scipy interpolation kind (default 'cubic')

    Returns
    -------
    envelope : np.ndarray of length `length`
    """
    from scipy.interpolate import interp1d

    if len(idx) < 2:
        return np.zeros(length)

    # Pad boundaries so the spline covers the full signal range
    x = np.concatenate([[0], idx, [length - 1]])
    y = np.concatenate([[val[0]], val, [val[-1]]])

    # Remove duplicate x values (can occur at boundaries)
    _, unique_mask = np.unique(x, return_index=True)
    x, y = x[unique_mask], y[unique_mask]

    f = interp1d(x, y, kind=kind if len(x) >= 4 else "linear", fill_value="extrapolate")
    return f(np.arange(length))


def _sift(signal: np.ndarray, max_iter: int = 100, sd_threshold: float = 0.2) -> np.ndarray:
    """
    Extract one IMF from `signal` via the sifting process.

    Stopping criterion: standard deviation between successive sifts < sd_threshold.

    Returns
    -------
    imf : np.ndarray  (same shape as signal)
    """
    h = signal.copy()

    for _ in range(max_iter):
        max_idx, max_val, min_idx, min_val = _find_extrema(h)

        if len(max_idx) < 2 or len(min_idx) < 2:
            break

        upper_env = _interpolate_envelope(max_idx, max_val, len(h))
        lower_env = _interpolate_envelope(min_idx, min_val, len(h))
        mean_env = (upper_env + lower_env) / 2.0

        h_prev = h.copy()
        h = h - mean_env

        # Cauchy-type stopping criterion
        sd = np.sum((h_prev - h) ** 2) / (np.sum(h_prev ** 2) + 1e-12)
        if sd < sd_threshold:
            break

    return h


def _emd(signal: np.ndarray, max_imfs: int = 10, sd_threshold: float = 0.2) -> List[np.ndarray]:
    """
    Standard EMD: decompose `signal` into a list of IMFs + residual.

    Parameters
    ----------
    signal       : 1-D time series
    max_imfs     : maximum number of IMFs to extract
    sd_threshold : sifting stopping criterion

    Returns
    -------
    imfs : list of np.ndarray  (last element is the residual)
    """
    residual = signal.copy()
    imfs: List[np.ndarray] = []

    for _ in range(max_imfs):
        # Stop if the residual is monotone (fewer than 2 extrema)
        max_idx, _, min_idx, _ = _find_extrema(residual)
        if len(max_idx) < 2 or len(min_idx) < 2:
            break

        imf = _sift(residual, sd_threshold=sd_threshold)
        imfs.append(imf)
        residual = residual - imf

    imfs.append(residual)  # final residual
    return imfs


# ---------------------------------------------------------------------------
# EEMD
# ---------------------------------------------------------------------------

class EEMDDecomposer:
    """
    Ensemble Empirical Mode Decomposition (EEMD).

    Parameters
    ----------
    ensemble_size : int
        Number of noise-added trials (N). Higher values reduce noise residue
        but increase computation time. Typical range: 50–200.
    noise_std : float
        Standard deviation of the added white noise (epsilon), expressed as a
        fraction of the signal's standard deviation. Typical value: 0.2.
    max_imfs : int
        Maximum number of IMFs to extract per trial.
    sd_threshold : float
        Sifting stopping criterion (standard deviation of successive sifts).
    random_seed : int or None
        Seed for reproducibility.

    Attributes
    ----------
    imfs_ : list of np.ndarray
        Averaged IMFs after fitting (includes residual as last element).
    n_imfs_ : int
        Total number of IMFs extracted (including residual).

    References
    ----------
    Wu, Z., & Huang, N. E. (2009). Ensemble empirical mode decomposition:
    A noise-assisted data analysis method. Advances in Adaptive Data Analysis.
    """

    def __init__(
        self,
        ensemble_size: int = 100,
        noise_std: float = 0.2,
        max_imfs: int = 10,
        sd_threshold: float = 0.2,
        random_seed: Optional[int] = 42,
    ):
        self.ensemble_size = ensemble_size
        self.noise_std = noise_std
        self.max_imfs = max_imfs
        self.sd_threshold = sd_threshold
        self.random_seed = random_seed

        self.imfs_: Optional[List[np.ndarray]] = None
        self.n_imfs_: Optional[int] = None
        self._signal_std: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, time_series: np.ndarray) -> "EEMDDecomposer":
        """
        Decompose `time_series` using EEMD.

        Parameters
        ----------
        time_series : np.ndarray, shape (T,)
            Univariate financial time series (e.g., log returns or price).

        Returns
        -------
        self
        """
        time_series = np.asarray(time_series, dtype=float)
        if time_series.ndim != 1:
            raise ValueError("time_series must be 1-D.")
        if len(time_series) < 10:
            raise ValueError("time_series is too short for EEMD (need ≥ 10 points).")

        self._signal_std = float(np.std(time_series))
        noise_amplitude = self.noise_std * self._signal_std

        rng = np.random.default_rng(self.random_seed)
        ensemble_imfs: List[List[np.ndarray]] = []

        for trial in range(self.ensemble_size):
            noise = rng.normal(0, noise_amplitude, size=len(time_series))
            noisy_signal = time_series + noise
            trial_imfs = _emd(noisy_signal, max_imfs=self.max_imfs, sd_threshold=self.sd_threshold)
            ensemble_imfs.append(trial_imfs)

        self.imfs_ = self._average_ensemble(ensemble_imfs, len(time_series))
        self.n_imfs_ = len(self.imfs_)
        return self

    def fit_transform(self, time_series: np.ndarray) -> List[np.ndarray]:
        """Fit and return the list of averaged IMFs."""
        return self.fit(time_series).imfs_

    def get_imf(self, index: int) -> np.ndarray:
        """Return a single IMF by index (0-based; last index = residual)."""
        self._check_fitted()
        if index < 0 or index >= self.n_imfs_:
            raise IndexError(f"IMF index {index} out of range [0, {self.n_imfs_ - 1}].")
        return self.imfs_[index]

    def reconstruct(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Reconstruct signal from selected IMFs.

        Parameters
        ----------
        indices : list of int or None
            Indices of IMFs to sum. If None, all IMFs are used (full reconstruction).

        Returns
        -------
        reconstructed : np.ndarray
        """
        self._check_fitted()
        if indices is None:
            indices = list(range(self.n_imfs_))
        return np.sum([self.imfs_[i] for i in indices], axis=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _average_ensemble(
        ensemble_imfs: List[List[np.ndarray]],
        signal_length: int,
    ) -> List[np.ndarray]:
        """
        Align IMF lists (which may differ in length across trials) and average.

        Strategy: use the maximum number of IMFs observed across trials; pad
        shorter trials with zeros for the missing high-order IMFs.
        """
        max_n = max(len(trial) for trial in ensemble_imfs)

        averaged: List[np.ndarray] = []
        for k in range(max_n):
            stack = []
            for trial in ensemble_imfs:
                if k < len(trial):
                    stack.append(trial[k])
                else:
                    # Trial had fewer IMFs — contribute zeros for this mode
                    stack.append(np.zeros(signal_length))
            averaged.append(np.mean(stack, axis=0))

        return averaged

    def _check_fitted(self):
        if self.imfs_ is None:
            raise RuntimeError("EEMDDecomposer has not been fitted yet. Call fit() first.")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EEMDDecomposer("
            f"ensemble_size={self.ensemble_size}, "
            f"noise_std={self.noise_std}, "
            f"max_imfs={self.max_imfs})"
        )
