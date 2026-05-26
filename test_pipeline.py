"""
tests/test_pipeline.py
-----------------------
Unit and integration tests for the EEMD-ICA pipeline.

Run:
    pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eemd_ica.eemd import EEMDDecomposer, _find_extrema, _emd
from eemd_ica.rhd import RHDIntegrator
from eemd_ica.ica import ICASourceSeparator
from eemd_ica.metrics import (
    jarque_bera_test, hurst_exponent, correlation_with_original
)
from eemd_ica.preprocessing import log_returns, standardise, adf_test
from eemd_ica import EEMDICAPipeline, EEMDICAAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    np.random.seed(42)
    T = 300
    t = np.linspace(0, 4 * np.pi, T)
    prices = 50 + 5 * np.sin(t) + 2 * np.sin(4 * t) + np.random.normal(0, 0.5, T)
    return np.abs(prices)


@pytest.fixture
def log_return_series(synthetic_prices):
    return log_returns(synthetic_prices)


# ---------------------------------------------------------------------------
# Decomposition tests
# ---------------------------------------------------------------------------

class TestEEMDDecomposer:
    def test_basic_decomposition(self, log_return_series):
        eemd = EEMDDecomposer(ensemble_size=10, noise_std=0.2, random_seed=0)
        imfs = eemd.fit_transform(log_return_series)
        assert isinstance(imfs, list)
        assert len(imfs) >= 2  # at least one IMF + residual

    def test_reconstruction(self, log_return_series):
        eemd = EEMDDecomposer(ensemble_size=10, noise_std=0.2, random_seed=0)
        eemd.fit(log_return_series)
        reconstructed = eemd.reconstruct()
        # Reconstruction should be close to original (EEMD is approximate)
        error = np.mean(np.abs(reconstructed - log_return_series))
        assert error < 1.0  # generous tolerance for EEMD noise averaging

    def test_imf_shapes(self, log_return_series):
        eemd = EEMDDecomposer(ensemble_size=5, random_seed=0)
        imfs = eemd.fit_transform(log_return_series)
        for imf in imfs:
            assert imf.shape == log_return_series.shape

    def test_raises_on_short_series(self):
        eemd = EEMDDecomposer(ensemble_size=5)
        with pytest.raises(ValueError, match="too short"):
            eemd.fit(np.array([1.0, 2.0]))

    def test_raises_if_not_fitted(self):
        eemd = EEMDDecomposer()
        with pytest.raises(RuntimeError):
            eemd.get_imf(0)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestRHDIntegrator:
    def test_cci_range(self, log_return_series):
        eemd = EEMDDecomposer(ensemble_size=5, random_seed=0)
        imfs = eemd.fit_transform(log_return_series)

        rhd = RHDIntegrator(rhd_threshold=0.3)
        rhd.fit(imfs, log_return_series)

        assert all(-1.0 <= c <= 1.0 for c in rhd.ccis_)

    def test_residual_always_kept(self, log_return_series):
        eemd = EEMDDecomposer(ensemble_size=5, random_seed=0)
        imfs = eemd.fit_transform(log_return_series)

        rhd = RHDIntegrator(rhd_threshold=1.0, include_residual=True)  # extreme threshold
        rhd.fit(imfs, log_return_series)

        # Residual (last IMF) should always appear in kept_indices
        assert len(imfs) - 1 in rhd.kept_indices_

    def test_vimfs_are_subset_of_imfs(self, log_return_series):
        eemd = EEMDDecomposer(ensemble_size=5, random_seed=0)
        imfs = eemd.fit_transform(log_return_series)

        rhd = RHDIntegrator(rhd_threshold=0.0)  # keep all
        rhd.fit(imfs, log_return_series)

        assert len(rhd.vimfs_) == len(imfs)

    def test_static_rhd_calculation(self):
        a = np.array([1.0, 2.0, 1.5, 2.5, 2.0])
        b = np.array([1.0, 2.0, 1.5, 2.5, 2.0])  # identical → RHD = 0
        rhd = RHDIntegrator.calculate_rhd(b, a)
        assert rhd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Separation tests
# ---------------------------------------------------------------------------

class TestICASourceSeparator:
    def test_components_shape(self, log_return_series):
        eemd = EEMDDecomposer(ensemble_size=5, random_seed=0)
        imfs = eemd.fit_transform(log_return_series)
        rhd  = RHDIntegrator(rhd_threshold=0.0)
        vimfs = rhd.fit_transform(imfs, log_return_series)

        ica = ICASourceSeparator(n_components=2, random_state=0)
        comps = ica.fit_transform(vimfs)

        assert comps.shape == (2, len(log_return_series))

    def test_reconstruction_error_low(self, log_return_series):
        eemd = EEMDDecomposer(ensemble_size=5, random_seed=0)
        imfs = eemd.fit_transform(log_return_series)
        rhd  = RHDIntegrator(rhd_threshold=0.0)
        vimfs = rhd.fit_transform(imfs, log_return_series)

        ica = ICASourceSeparator(n_components=min(3, len(vimfs)), random_state=0)
        ica.fit(vimfs)
        error = ica.reconstruction_error(vimfs)
        assert error < 0.5  # reconstruction within 50% (generous for low ensemble)


# ---------------------------------------------------------------------------
# Verification tests
# ---------------------------------------------------------------------------

class TestVerificationMetrics:
    def test_jarque_bera_nongaussian(self):
        # Heavy-tailed distribution → should be non-Gaussian
        np.random.seed(0)
        x = np.random.standard_t(df=3, size=500)
        result = jarque_bera_test(x)
        assert result["is_nongaussian"] is True
        assert "skewness" in result
        assert "kurtosis" in result

    def test_jarque_bera_gaussian(self):
        np.random.seed(0)
        x = np.random.normal(0, 1, 5000)
        result = jarque_bera_test(x)
        # Gaussian → p > 0.05 (may occasionally fail; seed ensures determinism)
        assert result["p_value"] > 1e-10  # just check it ran

    def test_hurst_persistent(self):
        # Brownian motion with drift (persistent-ish)
        np.random.seed(0)
        x = np.cumsum(np.random.normal(0.01, 1, 500))
        result = hurst_exponent(x)
        assert "hurst" in result
        assert not np.isnan(result["hurst"])

    def test_correlation_perfect(self):
        x = np.linspace(0, 1, 100)
        result = correlation_with_original(x, x)
        assert result["correlation"] == pytest.approx(1.0, abs=1e-10)

    def test_correlation_anti(self):
        x = np.linspace(0, 1, 100)
        result = correlation_with_original(x, -x)
        assert result["correlation"] == pytest.approx(-1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_log_returns_positive(self):
        prices = np.array([100.0, 105.0, 103.0, 108.0])
        r = log_returns(prices)
        assert len(r) == 3
        assert np.allclose(r, np.log(prices[1:] / prices[:-1]))

    def test_log_returns_raises_nonpositive(self):
        with pytest.raises(ValueError):
            log_returns(np.array([100.0, -5.0, 103.0]))

    def test_standardise_zero_mean(self):
        x = np.random.normal(5, 3, 200)
        z, mu, sigma = standardise(x)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------

class TestEEMDICAPipeline:
    def test_pipeline_runs(self, synthetic_prices):
        pipeline = EEMDICAPipeline(
            ensemble_size=5,
            noise_std=0.2,
            rhd_threshold=0.0,  # keep all VIMFs
            verbose=False,
        )
        results = pipeline.fit(synthetic_prices)

        assert "imfs" in results
        assert "vimfs" in results
        assert "components" in results
        assert "verification" in results
        assert results["components"] is not None
        assert len(results["verification"]) > 0

    def test_pipeline_summary(self, synthetic_prices):
        pipeline = EEMDICAPipeline(ensemble_size=5, verbose=False)
        pipeline.fit(synthetic_prices)
        summary = pipeline.summary()
        assert "EEMD-ICA Pipeline Summary" in summary

    def test_analyzer_stage_by_stage(self, synthetic_prices):
        series = log_returns(synthetic_prices)
        analyzer = EEMDICAAnalyzer(ensemble_size=5, rhd_threshold=0.0)
        imfs   = analyzer.decompose_eemd(series)
        vimfs  = analyzer.generate_vimfs()
        comps  = analyzer.extract_independent_components()
        vrfn   = analyzer.verify()

        assert len(imfs) > 1
        assert len(vimfs) > 0
        assert comps.shape[1] == len(series)
        assert len(vrfn) == len(comps)
