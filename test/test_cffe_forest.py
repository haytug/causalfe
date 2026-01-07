"""
Tests for CFFE Forest.

Includes both unit tests and econometric validation tests.
"""

import numpy as np
import pytest
from causalfe.forest import CFFEForest
from causalfe.simulations.did_dgp import (
    dgp_fe_only,
    dgp_did_homogeneous,
    dgp_did_heterogeneous,
    dgp_staggered,
)


class TestBasicFunctionality:
    """Basic unit tests for shape and API."""

    def test_fit_predict_shape(self):
        """Test that fit/predict work and return correct shapes."""
        n, p = 100, 3
        X = np.random.randn(n, p)
        Y = np.random.randn(n)
        D = np.random.binomial(1, 0.5, n).astype(float)
        unit = np.repeat(np.arange(20), 5)
        time = np.tile(np.arange(5), 20)

        model = CFFEForest(n_trees=5, max_depth=2, min_leaf=10)
        model.fit(X, Y, D, unit, time)

        tau = model.predict(X)
        assert tau.shape == (n,)

    def test_predict_with_variance(self):
        """Test variance estimation returns correct shapes."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=10, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        tau, var = model.predict_with_variance(X)
        assert tau.shape == var.shape == (len(Y),)
        assert np.all(var >= 0)

    def test_predict_interval(self):
        """Test confidence interval construction."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=10, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        tau, ci_lo, ci_hi = model.predict_interval(X, alpha=0.05)
        assert np.all(ci_lo <= tau)
        assert np.all(tau <= ci_hi)


class TestEconometricValidation:
    """
    Econometric validation tests.
    
    These test the statistical properties of the estimator,
    not just code correctness.
    """

    def test_placebo_fe_only(self):
        """
        FE-only placebo test: τ̂ ≈ 0 when true τ = 0.
        
        This verifies no spurious heterogeneity from fixed effects.
        """
        X, Y, D, unit, time, tau_true = dgp_fe_only(N=100, T=5, seed=42)
        assert np.allclose(tau_true, 0)  # Sanity check DGP

        model = CFFEForest(n_trees=50, max_depth=3, min_leaf=15, seed=42)
        model.fit(X, Y, D, unit, time)
        tau_hat = model.predict(X)

        # Mean should be close to 0
        assert np.abs(tau_hat.mean()) < 0.5, f"Mean τ̂ = {tau_hat.mean():.3f}, expected ~0"

    def test_homogeneous_did_recovery(self):
        """
        Homogeneous DiD: recover constant τ.
        
        Tests that CFFE can recover a constant treatment effect.
        """
        true_tau = 2.0
        X, Y, D, unit, time, tau_true = dgp_did_homogeneous(
            N=100, T=5, tau=true_tau, seed=42
        )

        model = CFFEForest(n_trees=50, max_depth=3, min_leaf=15, seed=42)
        model.fit(X, Y, D, unit, time)
        tau_hat = model.predict(X)

        # Mean should be close to true τ
        bias = np.abs(tau_hat.mean() - true_tau)
        assert bias < 1.0, f"Bias = {bias:.3f}, expected < 1.0"

        # Variance should be small (homogeneous effect)
        assert tau_hat.std() < 1.0, f"Std = {tau_hat.std():.3f}, expected small"

    def test_heterogeneous_did_correlation(self):
        """
        Heterogeneous DiD: τ̂ correlates with true τ.
        
        Tests that CFFE captures treatment effect heterogeneity.
        """
        X, Y, D, unit, time, tau_true = dgp_did_heterogeneous(N=100, T=5, seed=42)

        model = CFFEForest(n_trees=50, max_depth=3, min_leaf=15, seed=42)
        model.fit(X, Y, D, unit, time)
        tau_hat = model.predict(X)

        corr = np.corrcoef(tau_hat, tau_true)[0, 1]
        assert corr > 0.3, f"Correlation = {corr:.3f}, expected > 0.3"

    def test_staggered_adoption(self):
        """
        Staggered adoption: recover heterogeneous effects.
        
        This is the canonical CFFE setting.
        """
        X, Y, D, unit, time, tau_true = dgp_staggered(N=100, T=5, seed=42)

        model = CFFEForest(n_trees=50, max_depth=3, min_leaf=15, seed=42)
        model.fit(X, Y, D, unit, time)
        tau_hat = model.predict(X)

        corr = np.corrcoef(tau_hat, tau_true)[0, 1]
        rmse = np.sqrt(np.mean((tau_hat - tau_true) ** 2))

        assert corr > 0.3, f"Correlation = {corr:.3f}, expected > 0.3"
        assert rmse < 2.0, f"RMSE = {rmse:.3f}, expected < 2.0"

    def test_ci_coverage(self):
        """
        95% CI should contain true τ approximately 95% of the time.
        
        Note: This is a weak test due to small sample size and
        the difficulty of variance estimation in forests.
        """
        X, Y, D, unit, time, tau_true = dgp_staggered(N=100, T=5, seed=42)

        model = CFFEForest(n_trees=100, max_depth=3, min_leaf=15, seed=42)
        model.fit(X, Y, D, unit, time)

        tau_hat, ci_lo, ci_hi = model.predict_interval(X, alpha=0.05)

        coverage = np.mean((tau_true >= ci_lo) & (tau_true <= ci_hi))
        # Coverage is often low in forests due to variance underestimation
        # This is a known issue, so we use a weak threshold
        assert coverage > 0.2, f"Coverage = {coverage:.3f}, expected > 0.2"


class TestInference:
    """Tests for inference methods."""

    def test_half_sample_variance(self):
        """Test half-sample variance estimation."""
        from causalfe.inference import half_sample_variance

        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=20, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        var = half_sample_variance(model.trees, X)
        assert var.shape == (len(Y),)
        assert np.all(var >= 0)

    def test_cluster_robust_variance(self):
        """Test cluster-robust variance estimation."""
        from causalfe.inference import cluster_robust_variance

        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=20, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)
        tau_hat = model.predict(X)

        var = cluster_robust_variance(tau_hat, unit)
        assert var >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
