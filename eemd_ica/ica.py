"""
separation.py
-------------
Source Separation Layer: FastICA applied to VIMFs.

ICA model:
    X = A @ S

where:
    X  — observed VIMF matrix (n_vimfs × T)
    A  — mixing matrix (n_vimfs × n_components)
    S  — source (independent component) matrix (n_components × T)

After ICA, we recover:
    S  = W @ X    (W = A^{-1}, the unmixing matrix)

Transformation coefficients b_k:
    b_k = sum of the k-th column of the mixing matrix A.
    They represent the aggregate loading of each IC across all VIMFs,
    and are used to rank the economic importance of each factor.

Post-processing:
    - ICs are sign-normalised (positive dominant variance)
    - ICs are sorted by |b_k| in descending order (most important first)
"""

import numpy as np
from typing import List, Optional, Tuple
from sklearn.decomposition import FastICA
import warnings


class ICASourceSeparator:
    """
    FastICA-based source separator for VIMF matrices.

    Parameters
    ----------
    n_components : int or None
        Number of independent components to extract.
        If None, uses min(n_vimfs, T // 10) as a heuristic.
    max_iter : int
        Maximum number of FastICA iterations. Default: 1000.
    tol : float
        Convergence tolerance for FastICA. Default: 1e-4.
    fun : str
        Non-linearity for FastICA ('logcosh', 'exp', 'cube'). Default: 'logcosh'.
    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    components_ : np.ndarray, shape (n_components, T)
        Extracted independent components (ICs), sorted by |b_k| descending.
    mixing_matrix_ : np.ndarray, shape (n_vimfs, n_components)
        Estimated mixing matrix A.
    unmixing_matrix_ : np.ndarray, shape (n_components, n_vimfs)
        Estimated unmixing matrix W.
    transformation_coefficients_ : np.ndarray, shape (n_components,)
        b_k = column sum of A for each IC. Higher |b_k| → more influential factor.
    sort_order_ : np.ndarray, shape (n_components,)
        Indices that sort ICs by |b_k| descending.
    n_components_ : int
        Actual number of components extracted.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fun: str = "logcosh",
        random_state: Optional[int] = 42,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.fun = fun
        self.random_state = random_state

        # Fitted attributes
        self.components_: Optional[np.ndarray] = None
        self.mixing_matrix_: Optional[np.ndarray] = None
        self.unmixing_matrix_: Optional[np.ndarray] = None
        self.transformation_coefficients_: Optional[np.ndarray] = None
        self.sort_order_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None
        self._ica: Optional[FastICA] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, vimfs: List[np.ndarray]) -> "ICASourceSeparator":
        """
        Apply FastICA to a list of VIMFs.

        Parameters
        ----------
        vimfs : list of np.ndarray, each shape (T,)
            Variational IMFs produced by the Integration layer.

        Returns
        -------
        self
        """
        if len(vimfs) == 0:
            raise ValueError("vimfs list is empty. Check the Integration layer output.")

        # Build observation matrix X: shape (T, n_vimfs)
        X = np.column_stack([np.asarray(v, dtype=float) for v in vimfs])  # (T, n_vimfs)
        T, n_vimfs = X.shape

        # Determine number of components
        n_comp = self.n_components
        if n_comp is None:
            n_comp = max(1, min(n_vimfs, T // 10))
        if n_comp > n_vimfs:
            warnings.warn(
                f"n_components ({n_comp}) > n_vimfs ({n_vimfs}). "
                f"Clipping to {n_vimfs}.",
                UserWarning,
                stacklevel=2,
            )
            n_comp = n_vimfs

        self.n_components_ = n_comp

        # Run FastICA
        ica = FastICA(
            n_components=n_comp,
            max_iter=self.max_iter,
            tol=self.tol,
            fun=self.fun,
            random_state=self.random_state,
            whiten="unit-variance",
        )

        try:
            S = ica.fit_transform(X)  # (T, n_comp)  — source signals
        except Exception as exc:
            raise RuntimeError(f"FastICA failed: {exc}") from exc

        self._ica = ica

        # Mixing matrix A: shape (n_vimfs, n_comp)
        A = ica.mixing_  # sklearn attribute

        # Unmixing matrix W = components_ (before whitening correction in sklearn)
        W = ica.components_  # shape (n_comp, n_vimfs)

        # Transformation coefficients b_k = column sum of A
        b = A.sum(axis=0)  # shape (n_comp,)

        # Sort ICs by |b_k| descending (most influential first)
        sort_order = np.argsort(-np.abs(b))

        # Sign-normalise: ensure each IC has positive skewness
        S_sorted = S[:, sort_order].T  # (n_comp, T)
        for i in range(n_comp):
            if np.mean(S_sorted[i] ** 3) < 0:
                S_sorted[i] *= -1

        self.components_            = S_sorted          # (n_comp, T)
        self.mixing_matrix_         = A[:, sort_order]  # (n_vimfs, n_comp)
        self.unmixing_matrix_       = W[sort_order, :]  # (n_comp, n_vimfs)
        self.transformation_coefficients_ = b[sort_order]
        self.sort_order_            = sort_order

        return self

    def fit_transform(self, vimfs: List[np.ndarray]) -> np.ndarray:
        """
        Fit and return the independent components matrix.

        Returns
        -------
        components : np.ndarray, shape (n_components, T)
        """
        return self.fit(vimfs).components_

    def get_component(self, index: int) -> np.ndarray:
        """Return a single IC (0-based index, sorted by importance)."""
        self._check_fitted()
        if index < 0 or index >= self.n_components_:
            raise IndexError(f"Component index {index} out of range.")
        return self.components_[index]

    def reconstruction_error(self, vimfs: List[np.ndarray]) -> float:
        """
        Compute RMS reconstruction error: ||X - A @ S||_F / ||X||_F.

        A low value (< 0.05) indicates a faithful decomposition.
        """
        self._check_fitted()
        X = np.column_stack([np.asarray(v, dtype=float) for v in vimfs])
        X_hat = (self.mixing_matrix_ @ self.components_).T
        error = np.linalg.norm(X - X_hat, "fro") / (np.linalg.norm(X, "fro") + 1e-12)
        return float(error)

    def summary(self) -> str:
        """Human-readable summary of extracted ICs."""
        self._check_fitted()
        lines = ["ICA Source Separation Summary", "=" * 40]
        for k in range(self.n_components_):
            ic = self.components_[k]
            lines.append(
                f"  IC-{k+1:02d}  b_k={self.transformation_coefficients_[k]:+.4f}"
                f"  std={ic.std():.4f}  skew={self._skewness(ic):+.4f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        mu = x.mean()
        sigma = x.std()
        if sigma < 1e-12:
            return 0.0
        return float(np.mean(((x - mu) / sigma) ** 3))

    def _check_fitted(self):
        if self.components_ is None:
            raise RuntimeError("ICASourceSeparator has not been fitted yet.")

    def __repr__(self) -> str:
        return (
            f"ICASourceSeparator("
            f"n_components={self.n_components}, "
            f"fun='{self.fun}', "
            f"max_iter={self.max_iter})"
        )
