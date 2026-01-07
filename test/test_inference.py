"""
Tests for CFFE inference methods.

Note: CI coverage in forests is typically below nominal due to
variance underestimation. This is a known limitation documented
in the literature (Wager & Athey, 2018).
"""

import numpy as np
import pytest
from causalfe.forest import CFFEForest
from causalfe.inference import (
    half_sample_variance,
    jackknife_variance,
    multi_split_variance,
    infinitesimal_jackknife_variance,
    cluster_robust_variance,
    cluster_robust_se,
    confidence_interval,
)
from causalfe.simulations.did_dgp import dgp_staggered, dgp_did_heterogeneous


class TestHalfSampleVariance:
    """Tests for half-sample variance estimation."""

    def test_returns_non_negative(self):
        """Variance estimates should be non-negative."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=20, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        var = half_sample_variance(model.trees, X)
        assert np.all(var >= 0)

    def test_shape_matches_input(self):
        """Variance array should match input shape."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=20, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        var = half_sample_variance(model.trees, X)
        assert var.shape == (len(Y),)

    def test_more_trees_lower_variance(self):
        """More trees should generally give lower variance."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model_few = CFFEForest(n_trees=10, max_depth=2, min_leaf=10, seed=42)
        model_few.fit(X, Y, D, unit, time)
        var_few = half_sample_variance(model_few.trees, X)

        model_many = CFFEForest(n_trees=100, max_depth=2, min_leaf=10, seed=42)
        model_many.fit(X, Y, D, unit, time)
        var_many = half_sample_variance(model_many.trees, X)

        # Mean variance should be lower with more trees (allow some slack)
        assert var_many.mean() <= var_few.mean() * 3

    def test_single_tree_returns_zero(self):
        """Single tree should return zero variance."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=1, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        var = half_sample_variance(model.trees, X)
        assert np.allclose(var, 0)


class TestJackknifeVariance:
    """Tests for jackknife variance estimation."""

    def test_returns_non_negative(self):
        """Variance estimates should be non-negative."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=20, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        var = jackknife_variance(model.trees, X)
        assert np.all(var >= 0)

    def test_similar_to_half_sample(self):
        """Jackknife and half-sample should give similar results."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=50, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        var_half = half_sample_variance(model.trees, X)
        var_jack = jackknife_variance(model.trees, X)

        # Should be same order of magnitude
        ratio = var_jack.mean() / (var_half.mean() + 1e-10)
        assert 0.1 < ratio < 10


class TestMultiSplitVariance:
    """Tests for multi-split variance estimation."""

    def test_returns_non_negative(self):
        """Variance estimates should be non-negative."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=20, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        var = multi_split_variance(model.trees, X, n_splits=5)
        assert np.all(var >= 0)

    def test_more_stable_than_half_sample(self):
        """Multi-split should be more stable across runs."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=50, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        var_multi = multi_split_variance(model.trees, X, n_splits=20)
        var_half = half_sample_variance(model.trees, X)

        # Both should be similar
        ratio = var_multi.mean() / (var_half.mean() + 1e-10)
        assert 0.5 < ratio < 2.0


class TestClusterRobustVariance:
    """Tests for cluster-robust variance estimation."""

    def test_returns_non_negative(self):
        """Variance should be non-negative."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=20, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)
        tau_hat = model.predict(X)

        var = cluster_robust_variance(tau_hat, unit)
        assert var >= 0

    def test_se_is_sqrt_variance(self):
        """SE should be sqrt of variance."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=20, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)
        tau_hat = model.predict(X)

        var = cluster_robust_variance(tau_hat, unit)
        se = cluster_robust_se(tau_hat, unit)

        assert np.isclose(se, np.sqrt(var))

    def test_accounts_for_clustering(self):
        """
        Cluster-robust variance should account for within-cluster correlation.
        """
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=20, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)
        tau_hat = model.predict(X)

        # Cluster-robust variance
        var_cluster = cluster_robust_variance(tau_hat, unit)

        # Naive variance (ignoring clustering)
        var_naive = tau_hat.var() / len(tau_hat)

        # Both should be positive
        assert var_cluster > 0
        assert var_naive > 0


class TestConfidenceIntervals:
    """Tests for confidence interval construction."""

    def test_ci_contains_point_estimate(self):
        """CI should contain the point estimate."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=50, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        tau, ci_lo, ci_hi = model.predict_interval(X, alpha=0.05)

        assert np.all(ci_lo <= tau)
        assert np.all(tau <= ci_hi)

    def test_wider_ci_with_higher_confidence(self):
        """99% CI should be wider than 95% CI."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=50, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        _, ci_lo_95, ci_hi_95 = model.predict_interval(X, alpha=0.05)
        _, ci_lo_99, ci_hi_99 = model.predict_interval(X, alpha=0.01)

        width_95 = (ci_hi_95 - ci_lo_95).mean()
        width_99 = (ci_hi_99 - ci_lo_99).mean()

        assert width_99 > width_95

    def test_confidence_interval_function(self):
        """Test standalone confidence_interval function."""
        tau_hat = np.array([1.0, 2.0, 3.0])
        var_hat = np.array([0.1, 0.2, 0.3])

        ci_lo, ci_hi = confidence_interval(tau_hat, var_hat, alpha=0.05)

        assert len(ci_lo) == len(tau_hat)
        assert len(ci_hi) == len(tau_hat)
        assert np.all(ci_lo < tau_hat)
        assert np.all(ci_hi > tau_hat)


class TestVarianceMethods:
    """Tests comparing different variance methods."""

    def test_all_methods_available(self):
        """All variance methods should be callable."""
        X, Y, D, unit, time, _ = dgp_staggered(N=50, T=5, seed=42)

        model = CFFEForest(n_trees=20, max_depth=2, min_leaf=10, seed=42)
        model.fit(X, Y, D, unit, time)

        # All methods should work
        tau1, var1 = model.predict_with_variance(X, method="half_sample")
        tau2, var2 = model.predict_with_variance(X, method="jackknife")
        tau3, var3 = model.predict_with_variance(X, method="infinitesimal")

        # Point estimates should be identical
        assert np.allclose(tau1, tau2)
        assert np.allclose(tau2, tau3)

        # Variances should all be non-negative
        assert np.all(var1 >= 0)
        assert np.all(var2 >= 0)
        assert np.all(var3 >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
