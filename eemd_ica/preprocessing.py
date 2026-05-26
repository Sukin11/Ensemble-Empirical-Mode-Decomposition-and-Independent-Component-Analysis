"""
preprocessing.py
----------------
Data preprocessing utilities for financial time series.

Covers:
- Stationarity testing (ADF)
- Log-return transformation
- Structural break detection (Chow test)
- Normalisation / standardisation helpers
- Alignment and length validation for multi-series inputs
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
import warnings


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def log_returns(prices: np.ndarray, drop_na: bool = True) -> np.ndarray:
    """
    Compute log returns: r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices   : np.ndarray, shape (T,) — price series (must be positive)
    drop_na  : bool — if True, drop the first NaN element

    Returns
    -------
    returns : np.ndarray, shape (T,) or (T-1,)
    """
    prices = np.asarray(prices, dtype=float)
    if np.any(prices <= 0):
        raise ValueError("All price values must be positive for log returns.")
    r = np.log(prices[1:] / prices[:-1])
    return r if drop_na else np.concatenate([[np.nan], r])


def standardise(series: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Zero-mean, unit-variance standardisation.

    Returns
    -------
    (z-scored series, mean, std)
    """
    series = np.asarray(series, dtype=float)
    mu = series.mean()
    sigma = series.std()
    if sigma < 1e-12:
        warnings.warn("Series has near-zero variance. Returning unscaled.", UserWarning)
        return series - mu, mu, sigma
    return (series - mu) / sigma, mu, sigma


def unstandardise(series: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Reverse standardisation."""
    return series * sigma + mu


def rolling_normalise(series: np.ndarray, window: int = 252) -> np.ndarray:
    """
    Rolling z-score: (x_t - mean_{t-window:t}) / std_{t-window:t}.
    Useful for making non-stationary series locally stationary.
    """
    series = np.asarray(series, dtype=float)
    result = np.full_like(series, np.nan)
    for t in range(window, len(series)):
        window_data = series[t - window : t]
        mu = window_data.mean()
        sigma = window_data.std()
        result[t] = (series[t] - mu) / (sigma + 1e-12)
    return result


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------

def adf_test(series: np.ndarray, significance: float = 0.05) -> Dict[str, object]:
    """
    Augmented Dickey-Fuller test for stationarity.

    Requires statsmodels. Falls back gracefully if not installed.

    Parameters
    ----------
    series       : np.ndarray — time series to test
    significance : float      — significance level for rejection

    Returns
    -------
    dict with keys:
        'adf_statistic', 'p_value', 'n_lags', 'critical_values',
        'is_stationary' — bool (True if p_value < significance)
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        raise ImportError("Install statsmodels for ADF test: pip install statsmodels")

    series = np.asarray(series, dtype=float)
    result = adfuller(series, autolag="AIC")

    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "n_lags": int(result[2]),
        "critical_values": result[4],
        "is_stationary": bool(result[1] < significance),
    }


# ---------------------------------------------------------------------------
# Structural Break Detection (Chow Test)
# ---------------------------------------------------------------------------

def chow_test(
    series: np.ndarray,
    breakpoint: int,
) -> Dict[str, float]:
    """
    Chow test for a single structural break at `breakpoint`.

    Tests whether the regression parameters before and after `breakpoint`
    are equal (null: no structural break).

    Parameters
    ----------
    series     : np.ndarray — time series
    breakpoint : int        — index of the suspected break

    Returns
    -------
    dict with keys:
        'f_statistic', 'p_value', 'breakpoint',
        'has_break' — bool (True if p_value < 0.05)
    """
    from scipy import stats

    series = np.asarray(series, dtype=float)
    T = len(series)

    if breakpoint <= 2 or breakpoint >= T - 2:
        raise ValueError("breakpoint must be strictly inside [2, T-2].")

    y = series
    t = np.arange(T, dtype=float)

    def ols_rss(y_seg, t_seg):
        X = np.column_stack([np.ones_like(t_seg), t_seg])
        coef, rss, _, _ = np.linalg.lstsq(X, y_seg, rcond=None)
        if len(rss) == 0:
            pred = X @ coef
            rss = float(np.sum((y_seg - pred) ** 2))
        else:
            rss = float(rss[0])
        return rss

    rss_full = ols_rss(y, t)
    rss1 = ols_rss(y[:breakpoint], t[:breakpoint])
    rss2 = ols_rss(y[breakpoint:], t[breakpoint:])

    k = 2  # number of parameters (intercept + slope)
    numerator = (rss_full - rss1 - rss2) / k
    denominator = (rss1 + rss2) / (T - 2 * k)
    f_stat = numerator / (denominator + 1e-12)

    p_value = float(1 - stats.f.cdf(f_stat, k, T - 2 * k))

    return {
        "f_statistic": float(f_stat),
        "p_value": p_value,
        "breakpoint": breakpoint,
        "has_break": bool(p_value < 0.05),
    }


def find_structural_break(
    series: np.ndarray,
    search_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, object]:
    """
    Grid-search for the single structural break with the highest F-statistic.

    Parameters
    ----------
    series       : np.ndarray
    search_range : (start, end) indices to search; defaults to [T//5, 4T//5]

    Returns
    -------
    dict with keys:
        'best_breakpoint', 'f_statistic', 'p_value', 'has_break'
    """
    series = np.asarray(series, dtype=float)
    T = len(series)

    if search_range is None:
        lo, hi = T // 5, 4 * T // 5
    else:
        lo, hi = search_range

    best_f = -np.inf
    best_bp = lo

    for bp in range(lo, hi):
        result = chow_test(series, bp)
        if result["f_statistic"] > best_f:
            best_f = result["f_statistic"]
            best_result = result

    return {
        "best_breakpoint": best_result["breakpoint"],
        "f_statistic": best_result["f_statistic"],
        "p_value": best_result["p_value"],
        "has_break": best_result["has_break"],
    }


# ---------------------------------------------------------------------------
# Multi-series alignment
# ---------------------------------------------------------------------------

def align_series(*arrays: np.ndarray) -> List[np.ndarray]:
    """
    Trim all arrays to the length of the shortest one (right-aligned).
    Useful when economic proxy variables have different history lengths.
    """
    min_len = min(len(a) for a in arrays)
    return [np.asarray(a[-min_len:], dtype=float) for a in arrays]
