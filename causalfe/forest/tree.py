"""
Honest causal tree with fixed effects.

Key invariants:
- FE residualization is node-specific (uses only data in that node)
- Splits maximize τ-heterogeneity, not MSE
- Estimation is honest (split on one half, estimate on other)
"""

import numpy as np
from causalfe.forest.residuals import fe_residualize
from causalfe.forest.splitting import find_best_split, estimate_tau


class TreeNode:
    """Binary tree node for CFFE."""

    def __init__(self):
        self.is_leaf = True
        self.tau = 0.0
        self.n_samples = 0

        # Split info (for internal nodes)
        self.feature = None
        self.threshold = None
        self.gain = 0.0  # heterogeneity gain from the split (for feature importance)
        self.left = None
        self.right = None


class CFFETree:
    """
    Honest causal tree with fixed effects.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    min_leaf : int
        Minimum samples per leaf.
    honest : bool
        If True, use sample splitting for honest estimation.
    """

    def __init__(self, max_depth: int = 5, min_leaf: int = 20, honest: bool = True,
                 seed: int = None):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.honest = honest
        self.seed = seed
        self.root = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        D: np.ndarray,
        unit: np.ndarray,
        time: np.ndarray,
    ):
        """
        Fit the tree.

        For honest trees:
        - Split data into structure half and estimation half
        - Build tree structure on first half
        - Estimate τ in leaves using second half
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        D = np.asarray(D, dtype=float)
        unit = np.asarray(unit)
        time = np.asarray(time)

        n = X.shape[0]
        indices = np.arange(n)

        if self.honest:
            # Cluster-aware split: split by units, not observations.
            # Use a local seeded RNG so predictions are reproducible; relying on
            # the global np.random state made predict() non-deterministic.
            rng = np.random.default_rng(self.seed)
            unique_units = np.unique(unit)
            unique_units = rng.permutation(unique_units)
            half = len(unique_units) // 2

            struct_units = set(unique_units[:half])
            struct_mask = np.array([u in struct_units for u in unit])

            struct_idx = indices[struct_mask]
            est_idx = indices[~struct_mask]

            # Build structure
            self.root = self._build_tree(
                X[struct_idx],
                Y[struct_idx],
                D[struct_idx],
                unit[struct_idx],
                time[struct_idx],
                depth=0,
            )

            # Estimate τ in leaves using estimation sample
            self._estimate_leaves(
                self.root,
                X[est_idx],
                Y[est_idx],
                D[est_idx],
                unit[est_idx],
                time[est_idx],
            )
        else:
            self.root = self._build_tree(X, Y, D, unit, time, depth=0)

        return self

    def _build_tree(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        D: np.ndarray,
        unit: np.ndarray,
        time: np.ndarray,
        depth: int,
    ) -> TreeNode:
        """Recursively build tree structure."""
        node = TreeNode()
        node.n_samples = len(Y)

        # Node-specific FE residualization
        Y_tilde, D_tilde = fe_residualize(Y, D, unit, time)

        # Estimate τ for this node
        node.tau = estimate_tau(Y_tilde, D_tilde)

        # Check stopping conditions
        if depth >= self.max_depth:
            return node
        if len(Y) < 2 * self.min_leaf:
            return node

        # Find best split
        best_feat, best_thresh, best_score = find_best_split(
            X, Y_tilde, D_tilde, self.min_leaf
        )

        # No valid split found
        if best_feat < 0 or best_score <= 0:
            return node

        # Create split
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        node.is_leaf = False
        node.feature = best_feat
        node.threshold = best_thresh
        node.gain = best_score

        node.left = self._build_tree(
            X[left_mask],
            Y[left_mask],
            D[left_mask],
            unit[left_mask],
            time[left_mask],
            depth + 1,
        )

        node.right = self._build_tree(
            X[right_mask],
            Y[right_mask],
            D[right_mask],
            unit[right_mask],
            time[right_mask],
            depth + 1,
        )

        return node


    def _estimate_leaves(
        self,
        node: TreeNode,
        X: np.ndarray,
        Y: np.ndarray,
        D: np.ndarray,
        unit: np.ndarray,
        time: np.ndarray,
    ):
        """Re-estimate τ in leaves using estimation sample (for honesty)."""
        if len(Y) == 0:
            return

        if node.is_leaf:
            # Node-specific FE residualization on estimation sample
            Y_tilde, D_tilde = fe_residualize(Y, D, unit, time)
            node.tau = estimate_tau(Y_tilde, D_tilde)
            node.n_samples = len(Y)
            return

        # Route to children
        left_mask = X[:, node.feature] <= node.threshold
        right_mask = ~left_mask

        self._estimate_leaves(
            node.left,
            X[left_mask],
            Y[left_mask],
            D[left_mask],
            unit[left_mask],
            time[left_mask],
        )

        self._estimate_leaves(
            node.right,
            X[right_mask],
            Y[right_mask],
            D[right_mask],
            unit[right_mask],
            time[right_mask],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE for each observation."""
        X = np.asarray(X)
        n = X.shape[0]
        tau_hat = np.zeros(n)

        for i in range(n):
            tau_hat[i] = self._predict_one(self.root, X[i])

        return tau_hat

    def _predict_one(self, node: TreeNode, x: np.ndarray) -> float:
        """Predict for a single observation."""
        if node.is_leaf:
            return node.tau

        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)

    def get_leaf_indices(self, X: np.ndarray) -> np.ndarray:
        """Get leaf index for each observation (for variance estimation)."""
        X = np.asarray(X)
        n = X.shape[0]
        leaf_ids = np.zeros(n, dtype=int)

        for i in range(n):
            leaf_ids[i] = self._get_leaf_id(self.root, X[i], 0)

        return leaf_ids

    def _get_leaf_id(self, node: TreeNode, x: np.ndarray, current_id: int) -> int:
        """Get leaf ID for a single observation."""
        if node.is_leaf:
            return current_id

        if x[node.feature] <= node.threshold:
            return self._get_leaf_id(node.left, x, 2 * current_id + 1)
        else:
            return self._get_leaf_id(node.right, x, 2 * current_id + 2)

    def feature_importances(self, n_features: int) -> np.ndarray:
        """
        Sample-weighted heterogeneity-gain importance for each feature.

        Each internal node contributes (n_samples * gain) to the feature it
        split on, where gain is the tau-heterogeneity split score. This mirrors
        the impurity-decrease importance used by regression forests, but the
        "impurity" here is treatment-effect heterogeneity. Not normalized.
        """
        imp = np.zeros(n_features)

        def _walk(node):
            if node is None or node.is_leaf:
                return
            imp[node.feature] += node.n_samples * node.gain
            _walk(node.left)
            _walk(node.right)

        _walk(self.root)
        return imp
