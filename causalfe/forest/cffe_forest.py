"""
Causal Forest with Fixed Effects (CFFE).

Main estimator class that aggregates honest causal trees with:
- Cluster-aware subsampling (by unit)
- Node-specific FE residualization
- τ-heterogeneity splitting
"""

import numpy as np
from causalfe.forest.tree import CFFETree


class CFFEForest:
    """
    Causal Forest with Fixed Effects (CFFE).

    Parameters
    ----------
    n_trees : int
        Number of trees in the forest.
    max_depth : int
        Maximum depth of each tree.
    min_leaf : int
        Minimum samples per leaf.
    honest : bool
        If True, use honest estimation (sample splitting).
    subsample_ratio : float
        Fraction of units to subsample for each tree.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 5,
        min_leaf: int = 20,
        honest: bool = True,
        subsample_ratio: float = 0.5,
        seed: int = None,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.honest = honest
        self.subsample_ratio = subsample_ratio
        self.seed = seed
        self.trees = []
        self._rng = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        D: np.ndarray,
        unit: np.ndarray,
        time: np.ndarray,
    ):
        """
        Fit the CFFE forest.

        Parameters
        ----------
        X : array of shape (n, p)
            Covariates.
        Y : array of shape (n,)
            Outcome.
        D : array of shape (n,)
            Treatment indicator.
        unit : array of shape (n,)
            Unit identifiers.
        time : array of shape (n,)
            Time identifiers.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        D = np.asarray(D, dtype=float)
        unit = np.asarray(unit)
        time = np.asarray(time)

        self._rng = np.random.default_rng(self.seed)
        self.trees = []

        # Get unique units for cluster-aware subsampling
        unique_units = np.unique(unit)
        n_units = len(unique_units)
        n_subsample = max(1, int(n_units * self.subsample_ratio))

        for _ in range(self.n_trees):
            # Cluster-aware subsampling: sample units, not observations
            sampled_units = self._rng.choice(
                unique_units, size=n_subsample, replace=True
            )

            # Get all observations for sampled units
            mask = np.isin(unit, sampled_units)
            idx = np.where(mask)[0]

            # Build tree on subsample
            tree = CFFETree(
                max_depth=self.max_depth,
                min_leaf=self.min_leaf,
                honest=self.honest,
            )
            tree.fit(X[idx], Y[idx], D[idx], unit[idx], time[idx])
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE for each observation.

        Parameters
        ----------
        X : array of shape (n, p)
            Covariates.

        Returns
        -------
        tau_hat : array of shape (n,)
            Estimated CATEs.
        """
        X = np.asarray(X)
        n = X.shape[0]
        tau_sum = np.zeros(n)

        for tree in self.trees:
            tau_sum += tree.predict(X)

        return tau_sum / len(self.trees)


    def predict_with_variance(self, X: np.ndarray, method: str = "half_sample") -> tuple:
        """
        Predict CATE with variance estimate.

        Parameters
        ----------
        X : array of shape (n, p)
            Covariates.
        method : str
            Variance method: "half_sample", "jackknife", or "infinitesimal"

        Returns
        -------
        tau_hat : array of shape (n,)
            Estimated CATEs.
        var_hat : array of shape (n,)
            Variance estimates.
        """
        X = np.asarray(X)
        n = X.shape[0]
        B = len(self.trees)

        # Collect predictions from all trees
        preds = np.zeros((B, n))
        for b, tree in enumerate(self.trees):
            preds[b] = tree.predict(X)

        # Mean prediction
        tau_hat = preds.mean(axis=0)

        if method == "half_sample":
            # Half-sample variance (GRF-style)
            mid = B // 2
            tau_b1 = preds[:mid].mean(axis=0)
            tau_b2 = preds[mid : 2 * mid].mean(axis=0)
            var_hat = 0.5 * ((tau_b1 - tau_hat) ** 2 + (tau_b2 - tau_hat) ** 2)
            
        elif method == "jackknife":
            # Jackknife variance
            preds_sum = preds.sum(axis=0)
            var_hat = np.zeros(n)
            for b in range(B):
                tau_loo = (preds_sum - preds[b]) / (B - 1)
                var_hat += (tau_loo - tau_hat) ** 2
            var_hat *= (B - 1) / B
            
        elif method == "infinitesimal":
            # Infinitesimal jackknife
            var_hat = np.var(preds, axis=0, ddof=1) / B
            
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply small-sample correction (inflate variance for better coverage)
        # This is a heuristic that improves coverage in practice
        correction = B / (B - 2) if B > 2 else 1.0
        var_hat = var_hat * correction

        return tau_hat, var_hat

    def predict_interval(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> tuple:
        """
        Predict CATE with confidence intervals.

        Parameters
        ----------
        X : array of shape (n, p)
            Covariates.
        alpha : float
            Significance level (default 0.05 for 95% CI).

        Returns
        -------
        tau_hat : array of shape (n,)
            Estimated CATEs.
        ci_lower : array of shape (n,)
            Lower CI bound.
        ci_upper : array of shape (n,)
            Upper CI bound.
        """
        from scipy import stats

        tau_hat, var_hat = self.predict_with_variance(X)
        se = np.sqrt(var_hat)

        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower = tau_hat - z * se
        ci_upper = tau_hat + z * se

        return tau_hat, ci_lower, ci_upper
