# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [Unreleased]

### Planned
- C++ acceleration via pybind11 (optional)
- Dynamic treatment effects
- Event study integration
- Additional variance estimation methods
