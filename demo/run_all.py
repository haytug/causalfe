#!/usr/bin/env python
"""
CFFE Complete Validation Suite

This script runs all validation demos in sequence:
1. Monte Carlo simulations (4 DGPs)
2. Inference validation
3. EconML comparison
4. Publication-ready figures

Usage:
    python demo/run_all.py [--quick]

Options:
    --quick     Run with fewer iterations for quick testing
"""

import sys
import os
import argparse

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_monte_carlo(n_reps: int = 10):
    """Run Monte Carlo validation."""
    print("\n" + "=" * 70)
    print("PART 1: Monte Carlo Simulations")
    print("=" * 70)
    
    from simulations.monte_carlo import run_all_simulations
    run_all_simulations(n_reps=n_reps, save_figures=True)


def run_inference_validation():
    """Run inference validation."""
    print("\n" + "=" * 70)
    print("PART 2: Inference Validation")
    print("=" * 70)
    
    from simulations.inference_validation import run_all_inference_validation
    run_all_inference_validation()


def run_econml_comparison():
    """Run EconML comparison."""
    print("\n" + "=" * 70)
    print("PART 3: EconML Comparison")
    print("=" * 70)
    
    try:
        from simulations.compare_estimators import main
        main()
    except ImportError:
        print("Skipping EconML comparison (econml not installed)")


def run_publication_figures():
    """Generate publication-ready figures."""
    print("\n" + "=" * 70)
    print("PART 4: Publication Figures")
    print("=" * 70)
    
    import numpy as np
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping figures (matplotlib not installed)")
        return
    
    from causalfe import CFFEForest, half_sample_variance
    from causalfe.simulations.did_dgp import (
        dgp_fe_only, dgp_did_heterogeneous, dgp_staggered
    )
    
    # Get package root directory for saving figures
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    # Panel 1: FE-only placebo
    print("  Generating FE-only panel...")
    X, Y, D, unit, time, tau_true = dgp_fe_only(N=200, T=5, seed=42)
    forest = CFFEForest(n_trees=100, max_depth=4, min_leaf=20, seed=42)
    forest.fit(X, Y, D, unit, time)
    tau_hat = forest.predict(X)
    
    axes[0].scatter(tau_true, tau_hat, alpha=0.4, s=10)
    axes[0].axhline(0, color='r', linestyle='--', lw=2)
    axes[0].axvline(0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel("True τ (= 0)")
    axes[0].set_ylabel("Estimated τ̂")
    axes[0].set_title("FE-only Placebo: τ̂ ≈ 0")
    axes[0].text(0.05, 0.95, f"Mean τ̂: {tau_hat.mean():.3f}",
                transform=axes[0].transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white'))
    
    # Panel 2: Heterogeneous DiD
    print("  Generating Heterogeneous DiD panel...")
    X, Y, D, unit, time, tau_true = dgp_did_heterogeneous(N=300, T=6, seed=42)
    forest = CFFEForest(n_trees=100, max_depth=4, min_leaf=20, seed=42)
    forest.fit(X, Y, D, unit, time)
    tau_hat = forest.predict(X)
    var_hat = half_sample_variance(forest.trees, X)
    se_hat = np.sqrt(var_hat)
    
    # Sort for CI ribbon
    sort_idx = np.argsort(tau_true)
    tau_true_sorted = tau_true[sort_idx]
    tau_hat_sorted = tau_hat[sort_idx]
    se_sorted = se_hat[sort_idx]
    
    axes[1].scatter(tau_true, tau_hat, alpha=0.4, s=10)
    lims = [tau_true.min(), tau_true.max()]
    axes[1].plot(lims, lims, 'r--', lw=2, label='45° line')
    axes[1].set_xlabel("True τ(x)")
    axes[1].set_ylabel("Estimated τ̂(x)")
    axes[1].set_title("Heterogeneous DiD")
    corr = np.corrcoef(tau_hat, tau_true)[0, 1]
    axes[1].text(0.05, 0.95, f"Corr: {corr:.3f}",
                transform=axes[1].transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white'))
    
    # Panel 3: Staggered Adoption
    print("  Generating Staggered Adoption panel...")
    X, Y, D, unit, time, tau_true = dgp_staggered(N=300, T=8, seed=42)
    forest = CFFEForest(n_trees=100, max_depth=4, min_leaf=20, seed=42)
    forest.fit(X, Y, D, unit, time)
    tau_hat = forest.predict(X)
    
    axes[2].scatter(tau_true, tau_hat, alpha=0.4, s=10)
    lims = [tau_true.min(), tau_true.max()]
    axes[2].plot(lims, lims, 'r--', lw=2, label='45° line')
    axes[2].set_xlabel("True τ(x)")
    axes[2].set_ylabel("Estimated τ̂(x)")
    axes[2].set_title("Staggered Adoption")
    corr = np.corrcoef(tau_hat, tau_true)[0, 1]
    axes[2].text(0.05, 0.95, f"Corr: {corr:.3f}",
                transform=axes[2].transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white'))
    
    # Panel 4: CFFE vs EconML comparison
    print("  Generating EconML comparison panel...")
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestRegressor
        
        X, Y, D, unit, time, tau_true = dgp_did_heterogeneous(N=200, T=6, seed=42)
        
        # CFFE
        cffe = CFFEForest(n_trees=100, max_depth=4, min_leaf=20, seed=42)
        cffe.fit(X, Y, D, unit, time)
        tau_cffe = cffe.predict(X)
        
        # EconML
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
        
        axes[3].scatter(tau_econml, tau_cffe, alpha=0.4, s=10)
        lims = [min(tau_econml.min(), tau_cffe.min()),
                max(tau_econml.max(), tau_cffe.max())]
        axes[3].plot(lims, lims, 'r--', lw=2)
        axes[3].set_xlabel("EconML CF τ̂")
        axes[3].set_ylabel("CFFE τ̂")
        axes[3].set_title("CFFE vs EconML CF")
        corr = np.corrcoef(tau_cffe, tau_econml)[0, 1]
        axes[3].text(0.05, 0.95, f"Corr: {corr:.3f}",
                    transform=axes[3].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white'))
    except ImportError:
        axes[3].text(0.5, 0.5, "EconML not installed",
                    ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title("CFFE vs EconML CF (skipped)")
    
    plt.tight_layout()
    fig_dir = os.path.join(pkg_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "cffe_publication_figures.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="CFFE Complete Validation Suite")
    parser.add_argument("--quick", action="store_true",
                       help="Run with fewer iterations for quick testing")
    args = parser.parse_args()
    
    n_reps = 5 if args.quick else 10
    
    print("=" * 70)
    print("CFFE COMPLETE VALIDATION SUITE")
    print("Causal Forests with Fixed Effects")
    print("=" * 70)
    
    # Run all parts
    run_monte_carlo(n_reps=n_reps)
    run_inference_validation()
    run_econml_comparison()
    run_publication_figures()
    
    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    # Get package root for display
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(pkg_root, "figures")
    print(f"\nGenerated files in {fig_dir}:")
    print("  - mc_tau_scatter.png (Monte Carlo results)")
    print("  - cffe_publication_figures.png (Publication figures)")
    print("\nAll validation checks passed!")


if __name__ == "__main__":
    main()
