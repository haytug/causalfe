"""
Placebo tests for CFFE.

These tests verify that CFFE does not find spurious treatment effects
when none exist.
"""

import numpy as np
import pytest
from causalfe.forest import CFFEForest
from causalfe.simulations.did_dgp import dgp_fe_only


class TestPlacebo:
    """Placebo tests: τ̂ ≈ 0 when true τ = 0."""

    def test_fe_only_mean_near_zero(self):
        """Mean τ̂ should be near 0 in FE-only DGP."""
        X, Y, D, unit, time, tau_true = dgp_fe_only(N=100, T=5, seed=42)

        model = CFFEForest(n_trees=50, max_depth=3, min_leaf=15, seed=42)
        model.fit(X, Y, D, unit, time)
        tau_hat = model.predict(X)

        assert np.abs(tau_hat.mean()) < 0.5

    def test_fe_only_no_spurious_heterogeneity(self):
        """
        τ̂ should not show spurious heterogeneity in FE-only DGP.
        
        Standard CF would find spurious splits on unit/time.
        CFFE should not.
        """
        X, Y, D, unit, time, tau_true = dgp_fe_only(N=100, T=5, seed=42)

        model = CFFEForest(n_trees=50, max_depth=3, min_leaf=15, seed=42)
        model.fit(X, Y, D, unit, time)
        tau_hat = model.predict(X)

        # Variance of τ̂ should be small (no real heterogeneity)
        assert tau_hat.std() < 1.0

    def test_zero_treatment_effect(self):
        """
        When D has no effect on Y, τ̂ should be ~0.
        
        This is a direct test of the null hypothesis.
        """
        rng = np.random.default_rng(42)
        N, T = 100, 5
        n = N * T

        X = rng.normal(size=(n, 3))
        unit = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        # Staggered adoption but NO treatment effect
        adoption_time = rng.integers(1, T, size=N)
        D = (time >= adoption_time[unit]).astype(float)

        alpha = rng.normal(size=N)
        gamma = rng.normal(size=T)

        # Y is independent of D
        Y = alpha[unit] + gamma[time] + rng.normal(size=n)

        model = CFFEForest(n_trees=50, max_depth=3, min_leaf=15, seed=42)
        model.fit(X, Y, D, unit, time)
        tau_hat = model.predict(X)

        # Should be close to 0
        assert np.abs(tau_hat.mean()) < 0.5


class TestPlaceboMonteCarlo:
    """Monte Carlo placebo tests."""

    @pytest.mark.slow
    def test_placebo_mc(self):
        """
        Monte Carlo test: mean τ̂ should be ~0 across replications.
        """
        n_reps = 10
        means = []

        for seed in range(n_reps):
            X, Y, D, unit, time, _ = dgp_fe_only(N=50, T=5, seed=seed)

            model = CFFEForest(n_trees=30, max_depth=2, min_leaf=10, seed=seed)
            model.fit(X, Y, D, unit, time)
            tau_hat = model.predict(X)
            means.append(tau_hat.mean())

        # Average across replications should be ~0
        avg_mean = np.mean(means)
        assert np.abs(avg_mean) < 0.3, f"MC mean = {avg_mean:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
