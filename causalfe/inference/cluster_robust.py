"""
Cluster-robust variance estimation for CFFE.

Essential for valid inference in panel data where observations
are correlated within units.

Key principle: Resample units, not individual observations.
"""

import numpy as np
from typing import Callable, Optional


def cluster_robust_variance(
    tau_hat: np.ndarray,
    clusters: np.ndarray,
) -> float:
    """
    Cluster-robust variance estimator for the average treatment effect.

    Computes variance accounting for within-cluster correlation:

        V̂ = (1 / G(G-1)) * Σ_g (τ̄_g - τ̄)²

    where τ̄_g is the mean τ̂ within cluster g.

    Parameters
    ----------
    tau_hat : array of shape (n,)
        Estimated CATEs.
    clusters : array of shape (n,)
        Cluster (unit) identifiers.

    Returns
    -------
    var_hat : float
        Cluster-robust variance estimate for the ATE.

    Notes
    -----
    This estimates Var(ATE), not Var(CATE(x)).
    For observation-level variance, use cluster_bootstrap_variance.
    """
    tau_hat = np.asarray(tau_hat)
    clusters = np.asarray(clusters)

    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    if G < 2:
        return 0.0

    # Compute cluster means
    cluster_means = np.zeros(G)
    for g, c in enumerate(unique_clusters):
        mask = clusters == c
        cluster_means[g] = tau_hat[mask].mean()

    # Variance of cluster means
    grand_mean = cluster_means.mean()
    var_hat = np.sum((cluster_means - grand_mean) ** 2) / (G * (G - 1))

    return var_hat


def cluster_robust_se(tau_hat: np.ndarray, clusters: np.ndarray) -> float:
    """
    Cluster-robust standard error for the ATE.

    Parameters
    ----------
    tau_hat : array of shape (n,)
        Estimated CATEs.
    clusters : array of shape (n,)
        Cluster (unit) identifiers.

    Returns
    -------
    se : float
        Cluster-robust standard error.
    """
    return np.sqrt(cluster_robust_variance(tau_hat, clusters))


def cluster_bootstrap_variance(
    forest_factory: Callable,
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    B_boot: int = 200,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Cluster bootstrap variance estimator.

    Resamples units (not observations) and refits forest for each
    bootstrap iteration. This accounts for within-unit correlation.

    Algorithm:
    1. For b = 1, ..., B_boot:
       a. Sample units with replacement
       b. Get all observations for sampled units
       c. Fit new forest on bootstrap sample
       d. Predict τ̂_b(x) on original X
    2. Return Var(τ̂_b(x)) across bootstrap samples

    Parameters
    ----------
    forest_factory : callable
        Function that returns a new CFFEForest instance.
        Example: lambda: CFFEForest(n_trees=50, max_depth=3)
    X : array of shape (n, p)
        Covariates.
    Y : array of shape (n,)
        Outcome.
    D : array of shape (n,)
        Treatment.
    unit : array of shape (n,)
        Unit identifiers.
    time : array of shape (n,)
        Time identifiers.
    B_boot : int
        Number of bootstrap replications (default 200).
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print progress.

    Returns
    -------
    var_hat : array of shape (n,)
        Variance estimates for each observation.

    Notes
    -----
    - Slower than half-sample variance but more robust
    - Essential when half-sample variance is unreliable
    - Parallelizable (each bootstrap is independent)
    """
    rng = np.random.default_rng(seed)

    X = np.asarray(X)
    Y = np.asarray(Y)
    D = np.asarray(D, dtype=float)
    unit = np.asarray(unit)
    time = np.asarray(time)

    n = X.shape[0]
    unique_units = np.unique(unit)
    n_units = len(unique_units)

    # Collect bootstrap predictions
    boot_preds = np.zeros((B_boot, n))

    for b in range(B_boot):
        if verbose and (b + 1) % 50 == 0:
            print(f"  Bootstrap {b + 1}/{B_boot}")

        # Resample units with replacement
        boot_units = rng.choice(unique_units, size=n_units, replace=True)

        # Get observations for resampled units
        # Handle duplicates: if unit appears multiple times, include all its obs
        idx_list = []
        for u in boot_units:
            idx_list.extend(np.where(unit == u)[0].tolist())
        idx = np.array(idx_list)

        if len(idx) == 0:
            continue

        # Fit forest on bootstrap sample
        forest = forest_factory()
        forest.fit(X[idx], Y[idx], D[idx], unit[idx], time[idx])

        # Predict on original X
        boot_preds[b] = forest.predict(X)

    # Variance across bootstrap samples
    var_hat = boot_preds.var(axis=0, ddof=1)

    return var_hat


def wild_cluster_bootstrap_variance(
    forest: "CFFEForest",
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    B_boot: int = 200,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Wild cluster bootstrap variance estimator.

    Faster than full cluster bootstrap - perturbs residuals rather than
    refitting the entire forest.

    Parameters
    ----------
    forest : CFFEForest
        Already fitted forest.
    X, Y, D, unit, time : arrays
        Original data.
    B_boot : int
        Number of bootstrap replications.
    seed : int or None
        Random seed.

    Returns
    -------
    var_hat : array of shape (n,)
        Variance estimates.
    """
    rng = np.random.default_rng(seed)

    X = np.asarray(X)
    tau_hat = forest.predict(X)
    n = len(tau_hat)

    unique_units = np.unique(unit)

    # Wild bootstrap: perturb by cluster
    boot_preds = np.zeros((B_boot, n))

    for b in range(B_boot):
        # Rademacher weights by cluster
        weights = rng.choice([-1, 1], size=len(unique_units))
        unit_weights = dict(zip(unique_units, weights))

        # Apply weights
        perturbed = np.zeros(n)
        for i in range(n):
            perturbed[i] = tau_hat[i] * unit_weights[unit[i]]

        boot_preds[b] = perturbed

    var_hat = boot_preds.var(axis=0, ddof=1)

    return var_hat


def confidence_interval(
    tau_hat: np.ndarray,
    var_hat: np.ndarray,
    alpha: float = 0.05,
) -> tuple:
    """
    Construct confidence intervals using normal approximation.

    CI: τ̂(x) ± z_{1-α/2} * √V̂(x)

    Parameters
    ----------
    tau_hat : array of shape (n,)
        Point estimates.
    var_hat : array of shape (n,)
        Variance estimates.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    ci_lower, ci_upper : tuple of arrays
        Lower and upper confidence bounds.
    """
    from scipy import stats

    se = np.sqrt(np.maximum(var_hat, 0))  # Ensure non-negative
    z = stats.norm.ppf(1 - alpha / 2)

    ci_lower = tau_hat - z * se
    ci_upper = tau_hat + z * se

    return ci_lower, ci_upper
