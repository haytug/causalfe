# demo_cffe_readme.py
"""
One-command demo for causalfe:
- Runs 4 Monte Carlo simulations (FE-only, Homogeneous DiD, Heterogeneous DiD, Staggered)
- Fits CFFE forests
- Computes half-sample and cluster-robust variance
- Compares with EconML CF for FE-free DGP
- Plots results for publication/README
"""

import numpy as np
import matplotlib.pyplot as plt
from causalfe.forest import CFFEForest
from causalfe.simulations.did_dgp import (
    dgp_fe_only, dgp_did_homogeneous, dgp_did_heterogeneous, dgp_staggered
)
from causalfe.inference import half_sample_variance, cluster_robust_variance
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

def run_readme_demo():
    simulations = {
        "FE Only": dgp_fe_only,
        "Homogeneous DiD": dgp_did_homogeneous,
        "Heterogeneous DiD": dgp_did_heterogeneous,
        "Staggered Adoption": dgp_staggered
    }

    for name, dgp_func in simulations.items():
        print(f"\n=== Simulation: {name} ===")
        X, Y, D, unit, time, *tau_true = dgp_func()
        tau_true = tau_true[0] if tau_true else None

        # --- Fit CFFE forest ---
        forest = CFFEForest(n_trees=50, max_depth=3, min_leaf=10)
        forest.fit(X, Y, D, unit, time)
        tau_cffe = forest.predict(X)

        # --- Compute variance estimates ---
        V_half = half_sample_variance(forest.trees, X)
        V_cluster = cluster_robust_variance(tau_cffe, unit)  # scalar for ATE
        se_half = np.sqrt(V_half)
        se_cluster = np.sqrt(V_cluster)

        print(f"CFFE: mean τ̂={tau_cffe.mean():.3f}, "
              f"Half-sample SE={se_half.mean():.3f}, "
              f"Cluster-robust SE (ATE)={se_cluster:.3f}")

        # --- Compute RMSE / correlation / CI coverage if tau_true exists ---
        if tau_true is not None:
            rmse = np.sqrt(np.mean((tau_cffe - tau_true) ** 2))
            corr = np.corrcoef(tau_cffe, tau_true)[0,1]
            ci_lower = tau_cffe - 1.96 * se_half
            ci_upper = tau_cffe + 1.96 * se_half
            coverage = np.mean((tau_true >= ci_lower) & (tau_true <= ci_upper))

            print(f"RMSE: {rmse:.3f}, Corr: {corr:.3f}, 95% CI coverage: {coverage:.3f}")

            # Plot τ̂ vs true τ
            plt.figure(figsize=(5,5))
            plt.scatter(tau_true, tau_cffe, alpha=0.6)
            plt.plot([tau_true.min(), tau_true.max()],
                     [tau_true.min(), tau_true.max()],
                     'r--', lw=2)
            plt.fill_between(tau_true, ci_lower, ci_upper, color='gray', alpha=0.2)
            plt.xlabel("True τ(x)")
            plt.ylabel("CFFE τ̂(x)")
            plt.title(f"CFFE vs True τ: {name}")
            plt.show()

        # --- FE-only: Compare CFFE vs EconML CF ---
        if name == "FE Only":
            econml_cf = CausalForestDML(
                model_y=RandomForestRegressor(n_estimators=50, max_depth=3),
                model_t=RandomForestRegressor(n_estimators=50, max_depth=3),
                discrete_treatment=True,
                cv=2,
                n_estimators=50
            )
            econml_cf.fit(Y, D, X=X)
            tau_econml = econml_cf.effect(X)
            mean_diff = np.mean(tau_cffe - tau_econml)
            corr_cf = np.corrcoef(tau_cffe, tau_econml)[0,1]

            print(f"CFFE vs EconML CF mean diff: {mean_diff:.4f}, corr: {corr_cf:.4f}")
            # Scatter plot
            plt.figure(figsize=(5,5))
            plt.scatter(tau_econml, tau_cffe, alpha=0.6)
            plt.plot([tau_econml.min(), tau_econml.max()],
                     [tau_econml.min(), tau_econml.max()],
                     'r--', lw=2)
            plt.xlabel("EconML CF τ̂")
            plt.ylabel("CFFE τ̂")
            plt.title("FE-only: CFFE vs EconML CF")
            plt.show()


if __name__ == "__main__":
    run_readme_demo()
