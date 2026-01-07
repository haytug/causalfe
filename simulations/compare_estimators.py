"""
Compare CFFE vs Standard Causal Forest.

This script demonstrates that:
1. CFFE works with fixed effects
2. Standard CF fails with fixed effects
3. Standard CF produces spurious heterogeneity in FE-only DGP
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalfe.forest import CFFEForest
from simulations.monte_carlo import (
    dgp_fe_only,
    dgp_did_heterogeneous,
    dgp_staggered,
)

# Try to import EconML for comparison
try:
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False
    print("Warning: econml not installed. Skipping CF comparison.")


def run_cffe(X, Y, D, unit, time, seed=42):
    """Run CFFE."""
    forest = CFFEForest(n_trees=100, max_depth=4, min_leaf=20, seed=seed)
    forest.fit(X, Y, D, unit, time)
    return forest.predict(X)


def run_standard_cf(X, Y, D, seed=42):
    """Run standard causal forest (EconML)."""
    if not HAS_ECONML:
        return None
    
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=4, random_state=seed),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=4, random_state=seed),
        discrete_treatment=True,
        cv=2,
        n_estimators=100,
        random_state=seed,
    )
    cf.fit(Y, D, X=X)
    return cf.effect(X)


def compare_on_dgp(dgp_func, dgp_name, n_reps=5, **dgp_kwargs):
    """Compare CFFE vs Standard CF on a DGP."""
    print(f"\n{'=' * 60}")
    print(f"DGP: {dgp_name}")
    print("=" * 60)
    
    cffe_corrs = []
    cf_corrs = []
    cffe_rmses = []
    cf_rmses = []
    
    for rep in range(n_reps):
        print(f"  Rep {rep + 1}/{n_reps}", end="\r")
        
        X, Y, D, unit, time, tau_true = dgp_func(seed=rep, **dgp_kwargs)
        
        # CFFE
        tau_cffe = run_cffe(X, Y, D, unit, time, seed=rep)
        
        if tau_true.std() > 1e-10:
            cffe_corrs.append(np.corrcoef(tau_cffe, tau_true)[0, 1])
        cffe_rmses.append(np.sqrt(np.mean((tau_cffe - tau_true) ** 2)))
        
        # Standard CF
        if HAS_ECONML:
            tau_cf = run_standard_cf(X, Y, D, seed=rep)
            if tau_cf is not None:
                if tau_true.std() > 1e-10:
                    cf_corrs.append(np.corrcoef(tau_cf, tau_true)[0, 1])
                cf_rmses.append(np.sqrt(np.mean((tau_cf - tau_true) ** 2)))
    
    print()
    
    # Results
    print(f"\nCFFE:")
    print(f"  Mean RMSE: {np.mean(cffe_rmses):.3f}")
    if cffe_corrs:
        print(f"  Mean Corr: {np.mean(cffe_corrs):.3f}")
    
    if HAS_ECONML and cf_rmses:
        print(f"\nStandard CF (EconML):")
        print(f"  Mean RMSE: {np.mean(cf_rmses):.3f}")
        if cf_corrs:
            print(f"  Mean Corr: {np.mean(cf_corrs):.3f}")
    
    return {
        "cffe_rmse": np.mean(cffe_rmses),
        "cffe_corr": np.mean(cffe_corrs) if cffe_corrs else None,
        "cf_rmse": np.mean(cf_rmses) if cf_rmses else None,
        "cf_corr": np.mean(cf_corrs) if cf_corrs else None,
    }


def plot_comparison(dgp_func, dgp_name, seed=42, **dgp_kwargs):
    """Plot CFFE vs Standard CF scatter comparison."""
    X, Y, D, unit, time, tau_true = dgp_func(seed=seed, **dgp_kwargs)
    
    tau_cffe = run_cffe(X, Y, D, unit, time, seed=seed)
    
    fig, axes = plt.subplots(1, 2 if HAS_ECONML else 1, figsize=(12 if HAS_ECONML else 6, 5))
    
    if not HAS_ECONML:
        axes = [axes]
    
    # CFFE
    ax = axes[0]
    ax.scatter(tau_true, tau_cffe, alpha=0.3, s=10)
    lims = [min(tau_true.min(), tau_cffe.min()), max(tau_true.max(), tau_cffe.max())]
    ax.plot(lims, lims, "r--", lw=2)
    ax.set_xlabel("True τ(x)")
    ax.set_ylabel("Estimated τ̂(x)")
    ax.set_title(f"CFFE - {dgp_name}")
    
    corr = np.corrcoef(tau_cffe, tau_true)[0, 1] if tau_true.std() > 1e-10 else np.nan
    ax.text(0.05, 0.95, f"Corr: {corr:.3f}", transform=ax.transAxes, va="top")
    
    # Standard CF
    if HAS_ECONML:
        tau_cf = run_standard_cf(X, Y, D, seed=seed)
        ax = axes[1]
        ax.scatter(tau_true, tau_cf, alpha=0.3, s=10)
        lims = [min(tau_true.min(), tau_cf.min()), max(tau_true.max(), tau_cf.max())]
        ax.plot(lims, lims, "r--", lw=2)
        ax.set_xlabel("True τ(x)")
        ax.set_ylabel("Estimated τ̂(x)")
        ax.set_title(f"Standard CF - {dgp_name}")
        
        corr = np.corrcoef(tau_cf, tau_true)[0, 1] if tau_true.std() > 1e-10 else np.nan
        ax.text(0.05, 0.95, f"Corr: {corr:.3f}", transform=ax.transAxes, va="top")
    
    plt.tight_layout()
    return fig


def main():
    """Run all comparisons."""
    print("CFFE vs Standard CF Comparison")
    print("=" * 60)
    
    if not HAS_ECONML:
        print("\nNote: Install econml for full comparison:")
        print("  pip install econml")
    
    # Compare on each DGP
    results = {}
    
    results["fe_only"] = compare_on_dgp(
        dgp_fe_only, "FE-only (Placebo)", n_reps=5, N=200, T=5
    )
    
    results["heterogeneous"] = compare_on_dgp(
        dgp_did_heterogeneous, "Heterogeneous DiD", n_reps=5, N=300, T=6
    )
    
    results["staggered"] = compare_on_dgp(
        dgp_staggered, "Staggered Adoption", n_reps=5, N=300, T=8
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: CFFE vs Standard CF")
    print("=" * 60)
    print(f"{'DGP':<25} {'CFFE RMSE':>12} {'CF RMSE':>12} {'CFFE Corr':>12} {'CF Corr':>12}")
    print("-" * 75)
    
    for name, r in results.items():
        cf_rmse = f"{r['cf_rmse']:.3f}" if r['cf_rmse'] else "N/A"
        cf_corr = f"{r['cf_corr']:.3f}" if r['cf_corr'] else "N/A"
        cffe_corr = f"{r['cffe_corr']:.3f}" if r['cffe_corr'] else "N/A"
        print(f"{name:<25} {r['cffe_rmse']:>12.3f} {cf_rmse:>12} {cffe_corr:>12} {cf_corr:>12}")
    
    print("\nKey findings:")
    print("- CFFE handles fixed effects correctly")
    print("- Standard CF ignores panel structure")
    if HAS_ECONML:
        print("- Standard CF may show spurious heterogeneity in FE-only DGP")


if __name__ == "__main__":
    main()
