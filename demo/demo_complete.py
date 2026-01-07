#!/usr/bin/env python
"""
Complete CFFE Demo Script

This script demonstrates the full capabilities of the causalfe package:
1. Runs all 4 Monte Carlo simulations
2. Computes CATE estimates with variance
3. Compares CFFE vs EconML CF (backward compatibility)
4. Generates publication-ready figures

Usage:
    python demo/demo_complete.py
"""

import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalfe import CFFEForest, half_sample_variance, cluster_robust_variance

# Import DGPs from simulations module
from causalfe.simulations.did_dgp import (
    dgp_fe_only,
    dgp_did_homogeneous,
    dgp_did_heterogeneous,
    dgp_staggered,
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_simulation(name: str, dgp_func, n_trees: int = 100, **dgp_kwargs):
    """Run a single simulation and return results."""
    print(f"\n--- {name} ---")
    
    # Generate data
    X, Y, D, unit, time, tau_true = dgp_func(**dgp_kwargs)
    n = len(Y)
    
    # Fit CFFE
    forest = CFFEForest(
        n_trees=n_trees,
        max_depth=4,
        min_leaf=20,
        honest=True,
        seed=42,
    )
    forest.fit(X, Y, D, unit, time)
    
    # Predict with variance
    tau_hat, var_hat = forest.predict_with_variance(X)
    se_hat = np.sqrt(var_hat)
    
    # Compute metrics
    mean_tau = tau_hat.mean()
    rmse = np.sqrt(np.mean((tau_hat - tau_true) ** 2))
    
    # Correlation (only if tau_true has variation)
    if tau_true.std() > 1e-10:
        corr = np.corrcoef(tau_hat, tau_true)[0, 1]
    else:
        corr = np.nan
    
    # CI coverage
    ci_lo = tau_hat - 1.96 * se_hat
    ci_hi = tau_hat + 1.96 * se_hat
    coverage = np.mean((tau_true >= ci_lo) & (tau_true <= ci_hi))
    
    # Cluster-robust variance for ATE
    var_cluster = cluster_robust_variance(tau_hat, unit)
    
    # Print results
    print(f"  N = {n}, Mean τ̂ = {mean_tau:.3f}")
    print(f"  RMSE = {rmse:.3f}")
    if not np.isnan(corr):
        print(f"  Correlation = {corr:.3f}")
    print(f"  CI Coverage = {coverage:.3f}")
    print(f"  Cluster SE = {np.sqrt(var_cluster):.4f}")
    
    return {
        "name": name,
        "tau_hat": tau_hat,
        "tau_true": tau_true,
        "var_hat": var_hat,
        "mean_tau": mean_tau,
        "rmse": rmse,
        "corr": corr,
        "coverage": coverage,
    }


def run_econml_comparison():
    """Compare CFFE vs EconML CF on FE-free data."""
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        print("\nSkipping EconML comparison (econml not installed)")
        print("Install with: pip install econml")
        return None
    
    print_header("CFFE vs EconML CF Comparison (No Fixed Effects)")
    
    # Generate data without strong FE
    X, Y, D, unit, time, tau_true = dgp_did_heterogeneous(N=200, T=6, seed=42)
    
    # CFFE
    cffe = CFFEForest(n_trees=100, max_depth=4, min_leaf=20, seed=42)
    cffe.fit(X, Y, D, unit, time)
    tau_cffe = cffe.predict(X)
    
    # EconML CF
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
        discrete_treatment=True,
        cv=2,
        n_estimators=100,
        random_state=42,
    )
    cf.fit(Y, D, X=X)
    tau_econml = cf.effect(X)
    
    # Compare
    corr_cffe_econml = np.corrcoef(tau_cffe, tau_econml)[0, 1]
    corr_cffe_true = np.corrcoef(tau_cffe, tau_true)[0, 1]
    corr_econml_true = np.corrcoef(tau_econml, tau_true)[0, 1]
    
    print(f"\nCorrelation CFFE vs EconML: {corr_cffe_econml:.3f}")
    print(f"Correlation CFFE vs True:   {corr_cffe_true:.3f}")
    print(f"Correlation EconML vs True: {corr_econml_true:.3f}")
    
    if corr_cffe_econml > 0.8:
        print("\n✓ CFFE is backward-compatible with standard CF")
    
    return {
        "tau_cffe": tau_cffe,
        "tau_econml": tau_econml,
        "tau_true": tau_true,
        "corr_cffe_econml": corr_cffe_econml,
    }


def plot_results(results: list, comparison=None):
    """Generate publication-ready figures."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nSkipping plots (matplotlib not installed)")
        return
    
    print_header("Generating Figures")
    
    # Figure 1: τ̂ vs τ for all simulations
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, r in enumerate(results):
        ax = axes[i]
        ax.scatter(r["tau_true"], r["tau_hat"], alpha=0.3, s=10)
        
        lims = [
            min(r["tau_true"].min(), r["tau_hat"].min()),
            max(r["tau_true"].max(), r["tau_hat"].max()),
        ]
        ax.plot(lims, lims, "r--", lw=2, label="45° line")
        
        ax.set_xlabel("True τ(x)")
        ax.set_ylabel("Estimated τ̂(x)")
        ax.set_title(r["name"])
        
        # Add correlation text
        if not np.isnan(r["corr"]):
            ax.text(0.05, 0.95, f"Corr: {r['corr']:.3f}",
                   transform=ax.transAxes, va="top",
                   fontsize=10, bbox=dict(boxstyle="round", facecolor="white"))
    
    plt.tight_layout()
    # Save in figures directory
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "cffe_validation.png")
    plt.savefig(fig_path, dpi=150)
    print(f"  Saved: {fig_path}")
    plt.close()
    
    # Figure 2: CFFE vs EconML comparison
    if comparison is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(comparison["tau_econml"], comparison["tau_cffe"], alpha=0.3, s=10)
        
        lims = [
            min(comparison["tau_econml"].min(), comparison["tau_cffe"].min()),
            max(comparison["tau_econml"].max(), comparison["tau_cffe"].max()),
        ]
        ax.plot(lims, lims, "r--", lw=2)
        
        ax.set_xlabel("EconML CF τ̂(x)")
        ax.set_ylabel("CFFE τ̂(x)")
        ax.set_title("CFFE vs EconML CF (Backward Compatibility)")
        ax.text(0.05, 0.95, f"Corr: {comparison['corr_cffe_econml']:.3f}",
               transform=ax.transAxes, va="top",
               fontsize=10, bbox=dict(boxstyle="round", facecolor="white"))
        
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, "cffe_vs_econml.png")
        plt.savefig(fig_path, dpi=150)
        print(f"  Saved: {fig_path}")
        plt.close()


def main():
    """Run complete demo."""
    print_header("CFFE Complete Demo")
    print("Causal Forests with Fixed Effects")
    print("=" * 60)
    
    # Run all 4 simulations
    print_header("Monte Carlo Simulations")
    
    results = []
    
    results.append(run_simulation(
        "FE-only (Placebo)",
        dgp_fe_only,
        N=200, T=5, seed=42,
    ))
    
    results.append(run_simulation(
        "Homogeneous DiD (τ=2.0)",
        dgp_did_homogeneous,
        N=200, T=6, tau=2.0, seed=42,
    ))
    
    results.append(run_simulation(
        "Heterogeneous DiD",
        dgp_did_heterogeneous,
        N=300, T=6, seed=42,
    ))
    
    results.append(run_simulation(
        "Staggered Adoption",
        dgp_staggered,
        N=300, T=8, seed=42,
    ))
    
    # EconML comparison
    comparison = run_econml_comparison()
    
    # Summary table
    print_header("Summary Table")
    print(f"{'Simulation':<25} {'Mean τ̂':>10} {'RMSE':>10} {'Corr':>10} {'Coverage':>10}")
    print("-" * 65)
    for r in results:
        corr_str = f"{r['corr']:.3f}" if not np.isnan(r['corr']) else "N/A"
        print(f"{r['name']:<25} {r['mean_tau']:>10.3f} {r['rmse']:>10.3f} {corr_str:>10} {r['coverage']:>10.3f}")
    print("=" * 65)
    
    # Validation checks
    print_header("Validation Checks")
    
    checks = [
        ("FE-only: τ̂ ≈ 0", abs(results[0]["mean_tau"]) < 0.5),
        ("Homogeneous: τ̂ ≈ 2.0", abs(results[1]["mean_tau"] - 2.0) < 1.0),
        ("Heterogeneous: corr > 0.5", results[2]["corr"] > 0.5),
        ("Staggered: corr > 0.5", results[3]["corr"] > 0.5),
    ]
    
    all_pass = True
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed
    
    if all_pass:
        print("\n✓ All validation checks passed!")
    else:
        print("\n✗ Some checks failed")
    
    # Generate plots
    plot_results(results, comparison)
    
    print_header("Demo Complete")


if __name__ == "__main__":
    main()
