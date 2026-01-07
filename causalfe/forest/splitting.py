"""
FE-aware splitting for CFFE trees.

Core novelty: splits maximize τ-heterogeneity on residualized data.
"""

import numpy as np

# Try to import C++ core, fall back to pure Python
try:
    from causalfe.cpp import cffe_core

    USE_CPP = True
except ImportError:
    USE_CPP = False


def estimate_tau(Y_tilde: np.ndarray, D_tilde: np.ndarray) -> float:
    """
    IV-style CATE estimator: τ̂ = Σ D̃Ỹ / Σ D̃²
    """
    den = np.sum(D_tilde**2)
    if den < 1e-10:
        return 0.0
    return np.sum(D_tilde * Y_tilde) / den


def split_score(
    Y_tilde: np.ndarray, D_tilde: np.ndarray, left_mask: np.ndarray
) -> float:
    """
    τ-heterogeneity split score.
    Score = (nL * nR / n²) * (τL - τR)²
    """
    right_mask = ~left_mask
    nL, nR = left_mask.sum(), right_mask.sum()

    if nL == 0 or nR == 0:
        return 0.0

    tauL = estimate_tau(Y_tilde[left_mask], D_tilde[left_mask])
    tauR = estimate_tau(Y_tilde[right_mask], D_tilde[right_mask])

    n = len(Y_tilde)
    return (nL * nR / (n * n)) * (tauL - tauR) ** 2


def find_best_split(
    X: np.ndarray,
    Y_tilde: np.ndarray,
    D_tilde: np.ndarray,
    min_leaf: int,
) -> tuple:
    """
    Find best split across all features.

    Returns:
        (best_feature, best_threshold, best_score)
    """
    n, p = X.shape
    best_score = -1.0
    best_feature = -1
    best_threshold = 0.0

    for j in range(p):
        x_col = X[:, j]
        unique_vals = np.unique(x_col)

        if len(unique_vals) < 2:
            continue

        # Try midpoints between sorted unique values
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

        for thresh in thresholds:
            left_mask = x_col <= thresh
            nL, nR = left_mask.sum(), (~left_mask).sum()

            # Check min_leaf constraint
            if nL < min_leaf or nR < min_leaf:
                continue

            score = split_score(Y_tilde, D_tilde, left_mask)

            if score > best_score:
                best_score = score
                best_feature = j
                best_threshold = thresh

    return best_feature, best_threshold, best_score
