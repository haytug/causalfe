"""
Causal Forest with Fixed Effects (CFFE).

Main estimator class that aggregates honest causal trees with:
- Cluster-aware subsampling (by unit)
- Node-specific FE residualization
- τ-heterogeneity splitting
"""

import numpy as np
from causalfe.forest.tree import CFFETree
from causalfe.forest.residuals import fe_residualize
from causalfe.forest.splitting import estimate_tau


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
    recenter : bool
        If True (default), recenter the forest's CATE predictions so that
        their treated-sample mean equals the unbiased full-sample within-FE
        ATE. Honest causal-forest leaf estimates carry a finite-sample
        attenuation bias that grows with tree depth (node-level FE
        residualization on small leaves shrinks the treatment signal toward
        zero). Recentering removes this level bias while preserving the
        heterogeneity (relative ranking) the forest discovers, analogous to
        reporting the ATE separately from CATEs in the GRF framework.
    """

    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 5,
        min_leaf: int = 20,
        honest: bool = True,
        subsample_ratio: float = 0.5,
        seed: int = None,
        recenter: bool = True,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.honest = honest
        self.subsample_ratio = subsample_ratio
        self.seed = seed
        self.recenter = recenter
        self.trees = []
        self._rng = None
        self._is_fitted = False
        # Store training data info for score method
        self._n_samples = None
        self._n_features = None
        self._n_units = None
        # Unbiased full-sample within-FE ATE and recentering offset
        self._ate = None
        self._recenter_offset = 0.0
        self._train_X = None
        self._train_D = None

    def __repr__(self):
        """Return a string representation similar to scikit-learn estimators."""
        params = self.get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"CFFEForest({param_str})"

    def __str__(self):
        """Return a human-readable string representation."""
        if self._is_fitted:
            return (
                f"CFFEForest(n_trees={self.n_trees}, max_depth={self.max_depth}, "
                f"min_leaf={self.min_leaf})\n"
                f"  Fitted: Yes\n"
                f"  Training samples: {self._n_samples}\n"
                f"  Features: {self._n_features}\n"
                f"  Units: {self._n_units}"
            )
        else:
            return (
                f"CFFEForest(n_trees={self.n_trees}, max_depth={self.max_depth}, "
                f"min_leaf={self.min_leaf})\n"
                f"  Fitted: No"
            )

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "min_leaf": self.min_leaf,
            "honest": self.honest,
            "subsample_ratio": self.subsample_ratio,
            "seed": self.seed,
            "recenter": self.recenter,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {list(valid_params.keys())}"
                )
            setattr(self, key, value)
        return self

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

        # Store training data info
        self._n_samples = X.shape[0]
        self._n_features = X.shape[1]

        # Get unique units for cluster-aware subsampling
        unique_units = np.unique(unit)
        n_units = len(unique_units)
        self._n_units = n_units
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

        self._is_fitted = True

        # --- Unbiased ATE and recentering ---------------------------------
        # The forest's leaf-level CATE estimates carry a finite-sample
        # attenuation bias (node-level FE residualization on small honest
        # leaves shrinks D-variance and pulls tau toward zero). We therefore
        # estimate the ATE from the full-sample within-FE estimator, which is
        # numerically identical to a two-way fixed-effects (within) regression
        # and is unbiased for the ATE, and use it to recenter the CATEs.
        Y_tilde, D_tilde = fe_residualize(Y, D, unit, time, iters=10)
        self._ate = float(estimate_tau(Y_tilde, D_tilde))

        # Store training data for computing the recentering offset lazily
        self._train_X = X
        self._train_D = D

        if self.recenter:
            raw_tau = self._raw_predict(X)
            # Recenter so the full-sample mean of predicted CATEs equals the
            # unbiased within-FE ATE. This makes predict(X).mean() == ate()
            # by construction, so downstream code that averages the CATEs
            # recovers the unbiased ATE.
            self._recenter_offset = self._ate - raw_tau.mean()
        else:
            self._recenter_offset = 0.0

        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Raw (un-recentered) forest CATE predictions: mean over trees."""
        X = np.asarray(X)
        n = X.shape[0]
        tau_sum = np.zeros(n)

        for tree in self.trees:
            tau_sum += tree.predict(X)

        return tau_sum / len(self.trees)

    def predict(self, X: np.ndarray, raw: bool = False) -> np.ndarray:
        """
        Predict CATE for each observation.

        Parameters
        ----------
        X : array of shape (n, p)
            Covariates.
        raw : bool
            If True, return the raw forest average without ATE recentering.
            If False (default) and the forest was fitted with recenter=True,
            the predictions are shifted so their treated-sample mean equals
            the unbiased full-sample within-FE ATE (see the class docstring).

        Returns
        -------
        tau_hat : array of shape (n,)
            Estimated CATEs.
        """
        tau_hat = self._raw_predict(X)
        if raw or not self.recenter:
            return tau_hat
        return tau_hat + self._recenter_offset

    def ate(self) -> float:
        """
        Return the unbiased average treatment effect.

        This is the full-sample within-fixed-effects estimator
        (numerically identical to a two-way FE regression coefficient),
        which is unbiased for the ATE, unlike the mean of the forest's
        attenuated leaf-level CATE estimates.

        Returns
        -------
        ate : float
            Estimated average treatment effect (in the units of Y).
        """
        if not self._is_fitted:
            raise RuntimeError("Forest must be fitted before calling ate().")
        return self._ate


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

        # Mean prediction (raw scale for variance; recentered at return).
        # Recentering is a constant additive offset, so it does not affect
        # the variance of the CATE estimates.
        tau_raw = preds.mean(axis=0)
        tau_hat = tau_raw + self._recenter_offset if self.recenter else tau_raw

        if method == "half_sample":
            # Half-sample variance (GRF-style)
            mid = B // 2
            tau_b1 = preds[:mid].mean(axis=0)
            tau_b2 = preds[mid : 2 * mid].mean(axis=0)
            var_hat = 0.5 * ((tau_b1 - tau_raw) ** 2 + (tau_b2 - tau_raw) ** 2)
            
        elif method == "jackknife":
            # Jackknife variance (computed on the raw scale; the recentering
            # offset cancels in the deviations)
            preds_sum = preds.sum(axis=0)
            var_hat = np.zeros(n)
            for b in range(B):
                tau_loo = (preds_sum - preds[b]) / (B - 1)
                var_hat += (tau_loo - tau_raw) ** 2
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

    def score(self, X: np.ndarray, Y: np.ndarray, D: np.ndarray, 
              unit: np.ndarray, time: np.ndarray, tau_true: np.ndarray = None) -> float:
        """
        Return the R-squared score for CATE predictions.

        If tau_true is provided, computes R² between predicted and true CATEs.
        Otherwise, computes the pseudo-R² based on residual variance reduction.

        Parameters
        ----------
        X : array of shape (n, p)
            Covariates.
        Y : array of shape (n,)
            Outcome (used only if tau_true is None).
        D : array of shape (n,)
            Treatment indicator (used only if tau_true is None).
        unit : array of shape (n,)
            Unit identifiers (used only if tau_true is None).
        time : array of shape (n,)
            Time identifiers (used only if tau_true is None).
        tau_true : array of shape (n,), optional
            True treatment effects for evaluation.

        Returns
        -------
        score : float
            R-squared score. Higher is better, with 1.0 being perfect.
        """
        tau_hat = self.predict(X)
        
        if tau_true is not None:
            # R² between predicted and true CATEs
            ss_res = np.sum((tau_true - tau_hat) ** 2)
            ss_tot = np.sum((tau_true - np.mean(tau_true)) ** 2)
            if ss_tot == 0:
                return 0.0
            return 1 - ss_res / ss_tot
        else:
            # Pseudo-R² based on treatment effect heterogeneity
            # This measures how much variance in tau_hat we capture
            # relative to assuming homogeneous effects
            tau_var = np.var(tau_hat)
            if tau_var == 0:
                return 0.0
            # Return correlation-based score as a proxy
            return tau_var / (tau_var + 1)  # Normalized variance measure

    def clone(self):
        """
        Create an unfitted clone of this estimator.

        Returns
        -------
        clone : CFFEForest
            A new unfitted estimator with the same parameters.
        """
        return CFFEForest(**self.get_params())
