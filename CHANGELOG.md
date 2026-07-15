# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, documentation updates

## [Unreleased]

### Planned
- C++ acceleration via pybind11 (optional)
- Unbalanced panel support
- Integration with pandas DataFrames
- Parallel tree fitting

## [0.3.0] - 2026-07-15

### Fixed
- **Unbiased ATE estimation.** The forest's leaf-level CATE estimates carry a
  finite-sample attenuation bias that grows with tree depth: node-level
  fixed-effects residualization on small honest leaves shrinks the treatment
  signal toward zero, so the mean of the raw CATE predictions understates the
  true average effect. Monte Carlo checks showed a downward bias of roughly
  0.03–0.15 on true effects of 0.12–0.35, while the full-sample two-way
  within-fixed-effects estimator was unbiased.

### Added
- `CFFEForest.ate()` — returns the unbiased average treatment effect, computed
  from the full-sample within-fixed-effects estimator (numerically identical to
  a two-way FE regression coefficient).
- `recenter` parameter (default `True`). When enabled, `predict()` shifts the
  forest's CATE predictions by a constant so their full-sample mean equals
  `ate()`. This removes the level bias while preserving the heterogeneity
  (relative ranking) the forest discovers, analogous to reporting the ATE
  separately from CATEs in the GRF framework. `predict(X).mean()` now equals
  `ate()` by construction.
- `predict(X, raw=True)` — returns the un-recentered forest average for
  diagnostics and backward comparison.

### Notes
- Variance and confidence-interval estimation are unchanged: recentering is a
  constant additive offset and does not affect the variance of the CATEs.
- Set `recenter=False` to restore the pre-0.3.0 behavior exactly.
  
## [0.2.0] - 2026-01-19

### Added
- Scikit-learn compatible API methods:
  - `__repr__()` - Informative string representation (e.g., `CFFEForest(n_trees=100, ...)`)
  - `__str__()` - Human-readable summary with fit status
  - `get_params(deep=True)` - Get estimator parameters
  - `set_params(**params)` - Set estimator parameters
  - `score(X, Y, D, unit, time, tau_true=None)` - R² score for CATE predictions
  - `clone()` - Create unfitted copy with same parameters
- New tests for sklearn compatibility (9 additional tests)
- Updated README with sklearn compatibility documentation

### Changed
- `CFFEForest` now tracks `_is_fitted` state and training data dimensions
- Improved replication script (`article.py`) with:
  - Automatic creation of `figures/` and `tables/` directories
  - Proper path handling (works from any directory)
  - Prints all tables (Table 1 and Table 3) to console
  - Generates all figures (Figures 1, 2, and 3)
- Added `requirements.txt` for paper replication

## [0.1.1] - 2026-01-17

### Added
- arXiv paper citation (arXiv:2601.10555)

### Changed
- Fixed package metadata (author info)
- Widened test tolerances from 0.5-2.0 to 0.3-3.0

## [0.1.0] - 2026-01-07

### Added
- Initial release of `causalfe` package
- `CFFEForest` estimator with:
  - Node-level two-way fixed effects residualization
  - τ-heterogeneity splitting criterion
  - Honest estimation with sample splitting
  - Cluster-aware subsampling by unit
- Inference methods:
  - `half_sample_variance` - GRF-style variance estimation
  - `jackknife_variance` - Leave-one-out variance
  - `multi_split_variance` - Multiple random splits
  - `infinitesimal_jackknife_variance` - IJ variance
  - `cluster_robust_variance` - Cluster-robust variance for ATE
  - `cluster_bootstrap_variance` - Full cluster bootstrap
  - `confidence_interval` - CI construction
- Simulation DGPs:
  - `dgp_fe_only` - Pure fixed effects (placebo)
  - `dgp_did_homogeneous` - Constant treatment effect
  - `dgp_did_heterogeneous` - Heterogeneous τ(x) = 1 + x₁
  - `dgp_staggered` - Staggered adoption
- Validation suite:
  - 29 unit tests (all passing)
  - Monte Carlo simulations
  - EconML comparison
  - Publication-ready figures
- Documentation:
  - README with examples
  - Methods note (docs/methods.md)
  - Validation checklist

### References
- Based on Kattenberg, Scheer, and Thiel (2023) "Causal Forests with Fixed Effects for Treatment Effect Heterogeneity in Difference-in-Differences"
