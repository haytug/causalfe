"""
Half-sample variance estimation for CFFE.

This is the correct inference approach for honest forests with panel data.
Valid under:
- Clustered subsampling (by unit)
- Honest tree construction
- FE-orthogonal splitting

Reference: Athey et al. (2019), Wager & Athey (2018)
"""

import numpy as np


def half_sample_variance(trees: list, X: np.ndarray) -> np.ndarray:
    """
    Half-sample variance estimator (GRF-style).

    Splits the forest into two halves B1 and B2, computes forest estimates
    from each half, and estimates variance as:

        V̂(x) = (1/2) * [(τ̂_B1(x) - τ̂(x))² + (τ̂_B2(x) - τ̂(x))²]

    This is asymptotically valid under honesty and clustered subsampling.

    Parameters
    ----------
    trees : list of CFFETree
        Fitted trees from the forest.
    X : array of shape (n, p)
        Covariates for prediction.

    Returns
    -------
    var_hat : array of shape (n,)
        Variance estimates for each observation.

    Notes
    -----
    - Requires at least 2 trees
    - Valid under clustered sampling due to honesty
    - Fast: O(B * n) where B is number of trees
    """
    X = np.asarray(X)
    n = X.shape[0]
    B = len(trees)

    if B < 2:
        return np.zeros(n)

    # Collect predictions from all trees
    preds = np.zeros((B, n))
    for b, tree in enumerate(trees):
        preds[b] = tree.predict(X)

    # Overall forest mean: τ̂(x)
    tau_hat = preds.mean(axis=0)

    # Half-sample means: τ̂_B1(x) and τ̂_B2(x)
    mid = B // 2
    tau_b1 = preds[:mid].mean(axis=0)
    tau_b2 = preds[mid : 2 * mid].mean(axis=0)

    # Variance estimate: V̂(x) = (1/2) * [(τ̂_B1 - τ̂)² + (τ̂_B2 - τ̂)²]
    var_hat = 0.5 * ((tau_b1 - tau_hat) ** 2 + (tau_b2 - tau_hat) ** 2)

    return var_hat


def multi_split_variance(
    trees: list, X: np.ndarray, n_splits: int = 10
) -> np.ndarray:
    """
    Multi-split variance estimator.

    More stable than half-sample by averaging over multiple random splits.

    Parameters
    ----------
    trees : list of CFFETree
        Fitted trees from the forest.
    X : array of shape (n, p)
        Covariates for prediction.
    n_splits : int
        Number of random splits to average over.

    Returns
    -------
    var_hat : array of shape (n,)
        Variance estimates for each observation.
    """
    X = np.asarray(X)
    n = X.shape[0]
    B = len(trees)

    if B < 2:
        return np.zeros(n)

    # Collect predictions from all trees
    preds = np.zeros((B, n))
    for b, tree in enumerate(trees):
        preds[b] = tree.predict(X)

    tau_hat = preds.mean(axis=0)

    # Average variance over multiple random splits
    var_sum = np.zeros(n)
    rng = np.random.default_rng(42)

    for _ in range(n_splits):
        # Random permutation of trees
        perm = rng.permutation(B)
        mid = B // 2

        tau_b1 = preds[perm[:mid]].mean(axis=0)
        tau_b2 = preds[perm[mid : 2 * mid]].mean(axis=0)

        var_sum += 0.5 * ((tau_b1 - tau_hat) ** 2 + (tau_b2 - tau_hat) ** 2)

    return var_sum / n_splits


def jackknife_variance(trees: list, X: np.ndarray) -> np.ndarray:
    """
    Jackknife variance estimator (leave-one-tree-out).

    More stable but slower than half-sample. Computes:

        V̂(x) = ((B-1)/B) * Σ_b (τ̂_{-b}(x) - τ̂(x))²

    where τ̂_{-b} is the forest estimate excluding tree b.

    Parameters
    ----------
    trees : list of CFFETree
        Fitted trees from the forest.
    X : array of shape (n, p)
        Covariates for prediction.

    Returns
    -------
    var_hat : array of shape (n,)
        Variance estimates for each observation.
    """
    X = np.asarray(X)
    n = X.shape[0]
    B = len(trees)

    if B < 2:
        return np.zeros(n)

    # Collect predictions
    preds = np.zeros((B, n))
    for b, tree in enumerate(trees):
        preds[b] = tree.predict(X)

    tau_hat = preds.mean(axis=0)
    preds_sum = preds.sum(axis=0)

    # Leave-one-out variance
    var_hat = np.zeros(n)
    for b in range(B):
        # τ̂_{-b} = (sum - pred_b) / (B - 1)
        tau_loo = (preds_sum - preds[b]) / (B - 1)
        var_hat += (tau_loo - tau_hat) ** 2

    # Jackknife correction factor
    var_hat *= (B - 1) / B

    return var_hat


def infinitesimal_jackknife_variance(trees: list, X: np.ndarray) -> np.ndarray:
    """
    Infinitesimal jackknife variance estimator.

    Based on the variance of individual tree predictions, scaled appropriately.
    Faster than full jackknife for large forests.

    V̂(x) = (1/B²) * Σ_b (τ̂_b(x) - τ̂(x))²

    Parameters
    ----------
    trees : list of CFFETree
        Fitted trees from the forest.
    X : array of shape (n, p)
        Covariates for prediction.

    Returns
    -------
    var_hat : array of shape (n,)
        Variance estimates for each observation.
    """
    X = np.asarray(X)
    n = X.shape[0]
    B = len(trees)

    if B < 2:
        return np.zeros(n)

    # Collect predictions
    preds = np.zeros((B, n))
    for b, tree in enumerate(trees):
        preds[b] = tree.predict(X)

    tau_hat = preds.mean(axis=0)

    # Variance of tree predictions
    var_hat = np.mean((preds - tau_hat) ** 2, axis=0) / B

    return var_hat
