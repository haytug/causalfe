"""
Monte Carlo Validation for CFFE.

This script runs the 4 key simulations that validate CFFE:
1. FE-only (placebo) - τ̂ ≈ 0, no spurious splits
2. Homogeneous DiD - recover constant τ
3. Heterogeneous DiD - recover τ(x) = 1 + x₁
4. Staggered adoption - stress test

Run with: python -m simulations.monte_carlo
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Optional
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalfe.forest import CFFEForest
from causalfe.inference import half_sample_variance


# =============================================================================
# DGPs (exactly as specified)
# =============================================================================


def dgp_fe_only(N: int = 200, T: int = 5, seed: int = None):
    """
    Pure fixed effects - no treatment effect.
    
    Y_it = α_i + γ_t + ε_it
    
    Expected: CFFE should find no splits, τ̂ ≈ 0
    """
    rng = np.random.default_rng(seed)
    
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)
    
    alpha = rng.normal(size=N)
    gamma = rng.normal(size=T)
    
    Y = alpha[unit] + gamma[time] + rng.normal(size=N * T)
    
    # Staggered treatment for identification, but NO effect
    treat_time = rng.integers(2, T, size=N)
    D = (time >= treat_time[unit]).astype(float)
    
    X = rng.normal(size=(N * T, 2))
    
    tau_true = np.zeros(N * T)
    return X, Y, D, unit, time, tau_true


def dgp_did_homogeneous(N: int = 200, T: int = 6, tau: float = 2.0, seed: int = None):
    """
    Homogeneous DiD - constant treatment effect.
    
    Y_it = α_i + γ_t + τ·D_it + ε_it
    
    Expected: CFFE recovers τ̂ ≈ τ
    """
    rng = np.random.default_rng(seed)
    
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)
    
    alpha = rng.normal(size=N)
    gamma = rng.normal(size=T)
    
    # Treatment turns on at T/2 - need staggered for identification
    treat_time = rng.integers(2, T, size=N)
    D = (time >= treat_time[unit]).astype(float)
    
    Y = alpha[unit] + gamma[time] + tau * D + rng.normal(size=N * T)
    X = rng.normal(size=(N * T, 2))
    
    tau_true = np.full(N * T, tau)
    return X, Y, D, unit, time, tau_true


def dgp_did_heterogeneous(N: int = 300, T: int = 6, seed: int = None):
    """
    Heterogeneous DiD - treatment effect varies with X.
    
    τ(X) = 1 + X_{it,1}
    Y_it = α_i + γ_t + τ(X)·D_it + ε_it
    
    Expected: CFFE achieves corr(τ̂, τ) ≥ 0.5
    """
    rng = np.random.default_rng(seed)
    
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)
    
    X = rng.normal(size=(N * T, 2))
    tau_true = 1 + X[:, 0]  # Heterogeneous effect
    
    alpha = rng.normal(size=N)
    gamma = rng.normal(size=T)
    
    # Staggered adoption for identification
    treat_time = rng.integers(2, T, size=N)
    D = (time >= treat_time[unit]).astype(float)
    
    Y = alpha[unit] + gamma[time] + tau_true * D + rng.normal(size=N * T)
    
    return X, Y, D, unit, time, tau_true


def dgp_staggered(N: int = 300, T: int = 8, seed: int = None):
    """
    Staggered adoption - stress test.
    
    τ(X) = 1 + X_{it,1}
    Units adopt at different times.
    
    Expected: CFFE stable, attenuated; TWFE biased
    """
    rng = np.random.default_rng(seed)
    
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)
    
    X = rng.normal(size=(N * T, 2))
    tau_true = 1 + X[:, 0]
    
    alpha = rng.normal(size=N)
    gamma = rng.normal(size=T)
    
    # Staggered adoption: random treatment time per unit
    treat_time = rng.integers(3, T, size=N)
    D = (time >= treat_time[unit]).astype(float)
    
    Y = alpha[unit] + gamma[time] + tau_true * D + rng.normal(size=N * T)
    
    return X, Y, D, unit, time, tau_true


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    tau_hat: np.ndarray
    tau_true: np.ndarray
    var_hat: np.ndarray
    
    @property
    def mean_tau(self) -> float:
        return self.tau_hat.mean()
    
    @property
    def rmse(self) -> float:
        return np.sqrt(np.mean((self.tau_hat - self.tau_true) ** 2))
    
    @property
    def correlation(self) -> float:
        if self.tau_true.std() < 1e-10:
            return np.nan
        return np.corrcoef(self.tau_hat, self.tau_true)[0, 1]
    
    @property
    def ci_coverage(self) -> float:
        se = np.sqrt(self.var_hat)
        ci_lo = self.tau_hat - 1.96 * se
        ci_hi = self.tau_hat + 1.96 * se
        return np.mean((self.tau_true >= ci_lo) & (self.tau_true <= ci_hi))


def run_cffe(X, Y, D, unit, time, n_trees=100, max_depth=4, min_leaf=20, seed=None):
    """Run CFFE and return results."""
    forest = CFFEForest(
        n_trees=n_trees,
        max_depth=max_depth,
        min_leaf=min_leaf,
        seed=seed,
    )
    forest.fit(X, Y, D, unit, time)
    tau_hat = forest.predict(X)
    var_hat = half_sample_variance(forest.trees, X)
    return tau_hat, var_hat


# =============================================================================
# Monte Carlo Runner
# =============================================================================


def run_monte_carlo(
    dgp_func: Callable,
    n_reps: int = 20,
    n_trees: int = 100,
    max_depth: int = 4,
    min_leaf: int = 20,
    verbose: bool = True,
    **dgp_kwargs,
) -> list:
    """
    Run Monte Carlo simulation.
    
    Returns list of SimulationResult objects.
    """
    results = []
    
    for rep in range(n_reps):
        if verbose:
            print(f"  Rep {rep + 1}/{n_reps}", end="\r")
        
        # Generate data
        X, Y, D, unit, time, tau_true = dgp_func(seed=rep, **dgp_kwargs)
        
        # Run CFFE
        tau_hat, var_hat = run_cffe(
            X, Y, D, unit, time,
            n_trees=n_trees,
            max_depth=max_depth,
            min_leaf=min_leaf,
            seed=rep,
        )
        
        results.append(SimulationResult(tau_hat, tau_true, var_hat))
    
    if verbose:
        print()
    
    return results


def summarize_results(results: list, name: str):
    """Print summary statistics."""
    mean_tau = np.mean([r.mean_tau for r in results])
    std_tau = np.std([r.mean_tau for r in results])
    mean_rmse = np.mean([r.rmse for r in results])
    
    corrs = [r.correlation for r in results]
    corrs = [c for c in corrs if not np.isnan(c)]
    mean_corr = np.mean(corrs) if corrs else np.nan
    
    mean_coverage = np.mean([r.ci_coverage for r in results])
    
    print(f"\n{name}")
    print(f"  Mean τ̂:      {mean_tau:.3f} ± {std_tau:.3f}")
    print(f"  RMSE:        {mean_rmse:.3f}")
    print(f"  Correlation: {mean_corr:.3f}")
    print(f"  CI Coverage: {mean_coverage:.3f}")
    
    return {
        "mean_tau": mean_tau,
        "std_tau": std_tau,
        "rmse": mean_rmse,
        "correlation": mean_corr,
        "coverage": mean_coverage,
    }


# =============================================================================
# Plotting
# =============================================================================


def plot_tau_scatter(results: list, title: str, ax=None):
    """Plot τ̂ vs τ (scatter + 45° line)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Use last result for scatter
    r = results[-1]
    ax.scatter(r.tau_true, r.tau_hat, alpha=0.3, s=10)
    
    # 45° line
    lims = [
        min(r.tau_true.min(), r.tau_hat.min()),
        max(r.tau_true.max(), r.tau_hat.max()),
    ]
    ax.plot(lims, lims, "r--", lw=2, label="45° line")
    
    ax.set_xlabel("True τ(x)")
    ax.set_ylabel("Estimated τ̂(x)")
    ax.set_title(title)
    ax.legend()
    
    return ax


def plot_rmse_vs_n(dgp_func, N_values, n_reps=10, **dgp_kwargs):
    """Plot RMSE vs sample size."""
    rmses = []
    
    for N in N_values:
        print(f"  N = {N}", end="\r")
        results = run_monte_carlo(
            dgp_func, n_reps=n_reps, N=N, verbose=False, **dgp_kwargs
        )
        rmses.append(np.mean([r.rmse for r in results]))
    
    print()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_values, rmses, "o-", lw=2)
    ax.set_xlabel("N (units)")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs Sample Size")
    
    return fig, ax


def plot_coverage_vs_n(dgp_func, N_values, n_reps=10, **dgp_kwargs):
    """Plot CI coverage vs sample size."""
    coverages = []
    
    for N in N_values:
        print(f"  N = {N}", end="\r")
        results = run_monte_carlo(
            dgp_func, n_reps=n_reps, N=N, verbose=False, **dgp_kwargs
        )
        coverages.append(np.mean([r.ci_coverage for r in results]))
    
    print()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_values, coverages, "o-", lw=2)
    ax.axhline(0.95, color="r", linestyle="--", label="Nominal 95%")
    ax.set_xlabel("N (units)")
    ax.set_ylabel("CI Coverage")
    ax.set_title("CI Coverage vs Sample Size")
    ax.legend()
    ax.set_ylim(0, 1)
    
    return fig, ax


# =============================================================================
# Main Simulation Script
# =============================================================================


def run_all_simulations(n_reps: int = 20, save_figures: bool = True):
    """
    Run all 4 Monte Carlo simulations.
    
    This is the main validation script.
    """
    print("=" * 60)
    print("CFFE Monte Carlo Validation")
    print("=" * 60)
    
    all_results = {}
    
    # -------------------------------------------------------------------------
    # Simulation 1: FE-only (placebo)
    # -------------------------------------------------------------------------
    print("\n[1/4] FE-only (placebo)")
    print("Expected: τ̂ ≈ 0, no spurious heterogeneity")
    
    results_fe = run_monte_carlo(dgp_fe_only, n_reps=n_reps, N=200, T=5)
    stats_fe = summarize_results(results_fe, "FE-only Results")
    all_results["fe_only"] = (results_fe, stats_fe)
    
    # Check: mean τ̂ should be near 0
    if abs(stats_fe["mean_tau"]) < 0.5:
        print("  ✓ PASS: Mean τ̂ ≈ 0")
    else:
        print("  ✗ FAIL: Mean τ̂ not near 0")
    
    # -------------------------------------------------------------------------
    # Simulation 2: Homogeneous DiD
    # -------------------------------------------------------------------------
    print("\n[2/4] Homogeneous DiD (τ = 2.0)")
    print("Expected: τ̂ ≈ 2.0")
    
    results_hom = run_monte_carlo(
        dgp_did_homogeneous, n_reps=n_reps, N=200, T=6, tau=2.0
    )
    stats_hom = summarize_results(results_hom, "Homogeneous DiD Results")
    all_results["homogeneous"] = (results_hom, stats_hom)
    
    # Check: mean τ̂ should be near 2.0
    if abs(stats_hom["mean_tau"] - 2.0) < 1.0:
        print("  ✓ PASS: Mean τ̂ ≈ 2.0")
    else:
        print("  ✗ FAIL: Mean τ̂ not near 2.0")
    
    # -------------------------------------------------------------------------
    # Simulation 3: Heterogeneous DiD (key test)
    # -------------------------------------------------------------------------
    print("\n[3/4] Heterogeneous DiD (τ(x) = 1 + x₁)")
    print("Expected: corr(τ̂, τ) ≥ 0.5")
    
    results_het = run_monte_carlo(dgp_did_heterogeneous, n_reps=n_reps, N=300, T=6)
    stats_het = summarize_results(results_het, "Heterogeneous DiD Results")
    all_results["heterogeneous"] = (results_het, stats_het)
    
    # Check: correlation should be positive
    if stats_het["correlation"] > 0.3:
        print("  ✓ PASS: Correlation > 0.3")
    else:
        print("  ✗ FAIL: Correlation too low")
    
    # -------------------------------------------------------------------------
    # Simulation 4: Staggered adoption (stress test)
    # -------------------------------------------------------------------------
    print("\n[4/4] Staggered Adoption (stress test)")
    print("Expected: stable estimates, positive correlation")
    
    results_stag = run_monte_carlo(dgp_staggered, n_reps=n_reps, N=300, T=8)
    stats_stag = summarize_results(results_stag, "Staggered Adoption Results")
    all_results["staggered"] = (results_stag, stats_stag)
    
    # Check: correlation should be positive
    if stats_stag["correlation"] > 0.3:
        print("  ✓ PASS: Correlation > 0.3")
    else:
        print("  ✗ FAIL: Correlation too low")
    
    # -------------------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------------------
    if save_figures:
        print("\nGenerating figures...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        plot_tau_scatter(results_fe, "FE-only (Placebo)", axes[0, 0])
        plot_tau_scatter(results_hom, "Homogeneous DiD", axes[0, 1])
        plot_tau_scatter(results_het, "Heterogeneous DiD", axes[1, 0])
        plot_tau_scatter(results_stag, "Staggered Adoption", axes[1, 1])
        
        plt.tight_layout()
        # Save in figures directory
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fig_dir = os.path.join(pkg_root, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, "mc_tau_scatter.png")
        plt.savefig(fig_path, dpi=150)
        print(f"  Saved: {fig_path}")
        plt.close()
    
    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Simulation':<25} {'Mean τ̂':>10} {'RMSE':>10} {'Corr':>10} {'Coverage':>10}")
    print("-" * 65)
    print(f"{'FE-only (τ=0)':<25} {stats_fe['mean_tau']:>10.3f} {stats_fe['rmse']:>10.3f} {'N/A':>10} {stats_fe['coverage']:>10.3f}")
    print(f"{'Homogeneous (τ=2)':<25} {stats_hom['mean_tau']:>10.3f} {stats_hom['rmse']:>10.3f} {'N/A':>10} {stats_hom['coverage']:>10.3f}")
    print(f"{'Heterogeneous':<25} {stats_het['mean_tau']:>10.3f} {stats_het['rmse']:>10.3f} {stats_het['correlation']:>10.3f} {stats_het['coverage']:>10.3f}")
    print(f"{'Staggered':<25} {stats_stag['mean_tau']:>10.3f} {stats_stag['rmse']:>10.3f} {stats_stag['correlation']:>10.3f} {stats_stag['coverage']:>10.3f}")
    print("=" * 65)
    
    return all_results


if __name__ == "__main__":
    run_all_simulations(n_reps=20, save_figures=True)
