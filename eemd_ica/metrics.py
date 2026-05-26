"""
verification.py
---------------
Verification Layer: Statistical metrics to validate extracted ICs.

Four metrics are computed for each independent component (IC):

1. **Jarque-Bera (J-B) Test**
   Tests the null hypothesis of normality via skewness and excess kurtosis.
   ICA assumes non-Gaussianity, so significant non-normality (p < 0.05) is
   *required* for a valid IC.

2. **Hurst Exponent**
   Measures long-term memory via rescaled-range (R/S) analysis.
   H > 0.5 → persistent (trending) behaviour  ← expected for ICs
   H = 0.5 → random walk (no memory)
   H < 0.5 → anti-persistent (mean-reverting)

3. **Correlation Coefficient**
   Pearson correlation between each IC and the original price series.
   Ranks the individual contribution of each IC to overall price movement.

4. **Robust Regression (R²)**
   Regresses each IC against one or more economic proxy variables (e.g. GDP,
   exchange rates) using Huber M-estimator (robust to outliers).
   A significant R² confirms an economic interpretation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings

# Optional — only required for robust regression
try:
    from sklearn.linear_model import HuberRegressor
    _HUBER_AVAILABLE = True
except ImportError:
    _HUBER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def jarque_bera_test(ic: np.ndarray) -> Dict[str, float]:
    """
    Jarque-Bera normality test.

    Parameters
    ----------
    ic : np.ndarray — a single IC time series.

    Returns
    -------
    dict with keys:
        'statistic'  — JB test statistic
        'p_value'    — p-value (< 0.05 → reject normality)
        'skewness'   — sample skewness
        'kurtosis'   — excess kurtosis (Fisher's definition)
        'is_nongaussian' — bool, True if p_value < 0.05
    """
    ic = np.asarray(ic, dtype=float)
    jb_stat, p_value = stats.jarque_bera(ic)
    skew = float(stats.skew(ic))
    kurt = float(stats.kurtosis(ic))  # excess kurtosis

    return {
        "statistic": float(jb_stat),
        "p_value": float(p_value),
        "skewness": skew,
        "kurtosis": kurt,
        "is_nongaussian": bool(p_value < 0.05),
    }


def hurst_exponent(ic: np.ndarray, min_window: int = 8) -> Dict[str, float]:
    """
    Estimate the Hurst exponent via rescaled-range (R/S) analysis.

    The R/S statistic for a sub-series of length n is:
        R/S(n) = (max(cumdev) - min(cumdev)) / std(sub_series)

    Log-log regression of R/S vs n gives slope H (Hurst exponent).

    Parameters
    ----------
    ic         : np.ndarray — a single IC time series
    min_window : int        — minimum sub-series length (default 8)

    Returns
    -------
    dict with keys:
        'hurst'          — estimated Hurst exponent H
        'has_memory'     — bool, True if H > 0.5
        'interpretation' — string description
    """
    ic = np.asarray(ic, dtype=float)
    n_total = len(ic)

    if n_total < 2 * min_window:
        warnings.warn("Series too short for reliable Hurst estimation.", UserWarning)
        return {"hurst": float("nan"), "has_memory": False, "interpretation": "insufficient data"}

    # Window sizes as powers of 2 up to n_total / 2
    max_exp = int(np.log2(n_total // 2))
    min_exp = max(3, int(np.log2(min_window)))

    if max_exp <= min_exp:
        return {"hurst": float("nan"), "has_memory": False, "interpretation": "insufficient data"}

    ns = [2 ** exp for exp in range(min_exp, max_exp + 1)]
    rs_means: List[float] = []

    for n in ns:
        rs_vals = []
        for start in range(0, n_total - n + 1, n):
            sub = ic[start : start + n]
            mean = sub.mean()
            cumdev = np.cumsum(sub - mean)
            R = cumdev.max() - cumdev.min()
            S = sub.std(ddof=1)
            if S > 1e-12:
                rs_vals.append(R / S)
        if rs_vals:
            rs_means.append(np.mean(rs_vals))
        else:
            rs_means.append(float("nan"))

    # Filter out NaNs
    valid = [(np.log2(n), np.log2(rs)) for n, rs in zip(ns, rs_means) if not np.isnan(rs)]
    if len(valid) < 2:
        return {"hurst": float("nan"), "has_memory": False, "interpretation": "insufficient data"}

    log_n, log_rs = zip(*valid)
    slope, _, _, _, _ = stats.linregress(log_n, log_rs)
    H = float(slope)

    if H > 0.55:
        interp = "persistent (trending)"
    elif H < 0.45:
        interp = "anti-persistent (mean-reverting)"
    else:
        interp = "random walk"

    return {"hurst": H, "has_memory": bool(H > 0.5), "interpretation": interp}


def correlation_with_original(ic: np.ndarray, original: np.ndarray) -> Dict[str, float]:
    """
    Pearson correlation between an IC and the original price series.

    Parameters
    ----------
    ic       : np.ndarray — a single IC
    original : np.ndarray — original financial time series

    Returns
    -------
    dict with keys:
        'correlation' — Pearson r ∈ [-1, 1]
        'p_value'     — p-value of correlation test
        'abs_corr'    — |r| (used for ranking)
    """
    ic       = np.asarray(ic, dtype=float)
    original = np.asarray(original, dtype=float)

    if len(ic) != len(original):
        raise ValueError("ic and original must have the same length.")

    r, p = stats.pearsonr(ic, original)
    return {
        "correlation": float(r),
        "p_value": float(p),
        "abs_corr": float(abs(r)),
    }


def robust_regression(
    ic: np.ndarray,
    proxy_variables: np.ndarray,
    proxy_names: Optional[List[str]] = None,
    epsilon: float = 1.35,
) -> Dict[str, object]:
    """
    Robust regression of an IC on economic proxy variables (Huber M-estimator).

    Parameters
    ----------
    ic               : np.ndarray, shape (T,) — the IC to validate
    proxy_variables  : np.ndarray, shape (T, k) or (T,) — economic proxies
                       (e.g., GDP growth, exchange rate, inflation)
    proxy_names      : list of str or None — names for the proxy columns
    epsilon          : float — Huber epsilon (transition from quadratic to linear loss)

    Returns
    -------
    dict with keys:
        'r_squared'    — robust R² (1 - SS_res / SS_tot)
        'coefficients' — dict {name: coefficient}
        'intercept'    — regression intercept
        'is_significant' — bool, R² > 0.1 (heuristic threshold)
    """
    if not _HUBER_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for robust regression. "
            "Install with: pip install scikit-learn"
        )

    ic = np.asarray(ic, dtype=float)
    X = np.asarray(proxy_variables, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if len(ic) != len(X):
        raise ValueError("ic and proxy_variables must have the same number of rows.")

    if proxy_names is None:
        proxy_names = [f"proxy_{i+1}" for i in range(X.shape[1])]
    elif len(proxy_names) != X.shape[1]:
        raise ValueError("Length of proxy_names must match number of proxy columns.")

    model = HuberRegressor(epsilon=epsilon, max_iter=500)
    model.fit(X, ic)

    ic_pred = model.predict(X)
    ss_res = np.sum((ic - ic_pred) ** 2)
    ss_tot = np.sum((ic - ic.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return {
        "r_squared": float(r2),
        "coefficients": dict(zip(proxy_names, model.coef_.tolist())),
        "intercept": float(model.intercept_),
        "is_significant": bool(r2 > 0.1),
    }


# ---------------------------------------------------------------------------
# Aggregated verifier class
# ---------------------------------------------------------------------------

class ICVerifier:
    """
    Runs all four verification metrics on a set of independent components.

    Parameters
    ----------
    original : np.ndarray
        The original financial time series used to extract the ICs.
    proxy_variables : np.ndarray or None
        Economic proxy variables for robust regression (optional).
        Shape: (T, k) or (T,).
    proxy_names : list of str or None
        Column names for proxy_variables.

    Attributes
    ----------
    results_ : list of dict
        One dict per IC, containing results from all four metrics.
    """

    def __init__(
        self,
        original: np.ndarray,
        proxy_variables: Optional[np.ndarray] = None,
        proxy_names: Optional[List[str]] = None,
    ):
        self.original = np.asarray(original, dtype=float)
        self.proxy_variables = proxy_variables
        self.proxy_names = proxy_names
        self.results_: Optional[List[Dict]] = None

    def verify(self, components: np.ndarray) -> List[Dict]:
        """
        Verify each IC (row of `components`).

        Parameters
        ----------
        components : np.ndarray, shape (n_components, T)
            Independent components from ICASourceSeparator.

        Returns
        -------
        results : list of dict  (one per IC)
        """
        components = np.asarray(components, dtype=float)
        self.results_ = []

        for k, ic in enumerate(components):
            res: Dict = {"ic_index": k + 1}

            # 1. Jarque-Bera
            res["jarque_bera"] = jarque_bera_test(ic)

            # 2. Hurst
            res["hurst"] = hurst_exponent(ic)

            # 3. Correlation
            res["correlation"] = correlation_with_original(ic, self.original)

            # 4. Robust regression (if proxies provided)
            if self.proxy_variables is not None:
                try:
                    res["robust_regression"] = robust_regression(
                        ic, self.proxy_variables, self.proxy_names
                    )
                except Exception as exc:
                    res["robust_regression"] = {"error": str(exc)}
            else:
                res["robust_regression"] = None

            self.results_.append(res)

        return self.results_

    def summary(self) -> str:
        """Tabular summary of verification results."""
        if self.results_ is None:
            raise RuntimeError("Call verify() first.")

        header = (
            f"{'IC':>4} | {'JB p-val':>10} {'Non-Gauss':>10} "
            f"| {'Hurst':>7} {'Memory':>7} "
            f"| {'Corr':>7} "
            f"| {'R²':>7} {'Sig':>5}"
        )
        sep = "-" * len(header)
        lines = ["IC Verification Summary", "=" * len(header), header, sep]

        for res in self.results_:
            jb  = res["jarque_bera"]
            hu  = res["hurst"]
            co  = res["correlation"]
            rr  = res.get("robust_regression") or {}

            r2_str  = f"{rr.get('r_squared', float('nan')):7.4f}" if isinstance(rr, dict) and "r_squared" in rr else "    N/A"
            sig_str = str(rr.get("is_significant", "N/A")) if isinstance(rr, dict) else "N/A"

            lines.append(
                f"{res['ic_index']:>4} | "
                f"{jb['p_value']:>10.2e} {str(jb['is_nongaussian']):>10} "
                f"| {hu['hurst']:>7.4f} {str(hu['has_memory']):>7} "
                f"| {co['correlation']:>7.4f} "
                f"| {r2_str} {sig_str:>5}"
            )

        return "\n".join(lines)
