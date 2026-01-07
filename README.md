# causalfe

**Causal Forests with Fixed Effects in Python**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`causalfe` provides the first fully Pythonic implementation of Causal Forests with Fixed Effects (CFFE), enabling researchers and practitioners to estimate heterogeneous treatment effects in panel and difference-in-differences settings while rigorously controlling for unit and time fixed effects.

This package is a Python implementation inspired by [Kattenberg, Scheer, and Thiel (2023)](https://www.cpb.nl/), who developed the CFFE methodology and released an R package. We built this Python version to make CFFE accessible to the broader Python econometrics community.

### Key Features

- **Node-level FE residualization**: Fixed effects are removed within each tree node, not globally
- **τ-heterogeneity splitting**: Splits maximize treatment effect heterogeneity, not outcome variance
- **Honest estimation**: Separate samples for tree structure and leaf estimation
- **Cluster-aware inference**: Valid standard errors for panel data
- **Backward compatible**: Reduces to standard causal forest when no fixed effects present

## Installation

```bash
git clone https://github.com/haytug/causalfe.git
cd causalfe
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

For EconML comparison:
```bash
pip install -e ".[compare]"
```

## Quick Start

```python
from causalfe import CFFEForest

# Your panel data
# X: covariates (n, p)
# Y: outcome (n,)
# D: treatment (n,)
# unit: unit identifiers (n,)
# time: time identifiers (n,)

forest = CFFEForest(n_trees=100, max_depth=5, min_leaf=20)
forest.fit(X, Y, D, unit, time)

# Point estimates
tau_hat = forest.predict(X)

# With confidence intervals
tau_hat, ci_lower, ci_upper = forest.predict_interval(X, alpha=0.05)
```

## Example with Simulated Data

```python
from causalfe import CFFEForest
from causalfe.simulations.did_dgp import dgp_did_heterogeneous
import numpy as np

# Generate heterogeneous DiD data
X, Y, D, unit, time, tau_true = dgp_did_heterogeneous(N=200, T=6)

# Fit CFFE
forest = CFFEForest(n_trees=100, max_depth=4, min_leaf=20)
forest.fit(X, Y, D, unit, time)
tau_hat = forest.predict(X)

# Evaluate
corr = np.corrcoef(tau_hat, tau_true)[0, 1]
print(f"Correlation with true τ: {corr:.3f}")  # ~0.9
```

## Validation Results

| Simulation | Mean τ̂ | RMSE | Corr(τ̂, τ) | Status |
|------------|---------|------|-------------|--------|
| FE-only (τ=0) | ~0 | ~0.4 | N/A | ✓ |
| Homogeneous (τ=2) | ~1.8 | ~0.4 | N/A | ✓ |
| Heterogeneous DiD | varies | ~0.5 | **0.93** | ✓ |
| Staggered Adoption | varies | ~0.6 | **0.88** | ✓ |

## Inference

Multiple variance estimation methods are available:

```python
from causalfe import half_sample_variance, cluster_robust_variance

# Half-sample variance (fast, default)
tau_hat, var_hat = forest.predict_with_variance(X)

# Or use standalone functions
var_half = half_sample_variance(forest.trees, X)

# Cluster-robust variance for ATE
var_cluster = cluster_robust_variance(tau_hat, unit)
```

## API Reference

### CFFEForest

```python
CFFEForest(
    n_trees=100,      # Number of trees
    max_depth=5,      # Maximum tree depth
    min_leaf=20,      # Minimum samples per leaf
    honest=True,      # Use honest estimation
    subsample_ratio=0.5,  # Fraction of units to subsample
    seed=None,        # Random seed
)
```

**Methods:**
- `fit(X, Y, D, unit, time)`: Fit the forest
- `predict(X)`: Predict CATEs
- `predict_with_variance(X, method="half_sample")`: Predict with variance
- `predict_interval(X, alpha=0.05)`: Predict with confidence intervals

### Variance Functions

- `half_sample_variance(trees, X)`: Fast half-sample variance
- `jackknife_variance(trees, X)`: More stable jackknife variance
- `cluster_robust_variance(tau_hat, clusters)`: Cluster-robust variance
- `cluster_bootstrap_variance(...)`: Full cluster bootstrap

## Methodology

CFFE modifies the standard causal forest in two key ways:

1. **Node-level FE orthogonalization**: Within each node, we residualize Y and D:
   - Ỹ = Y - α̂ᵢ - γ̂ₜ
   - D̃ = D - α̂ᴰᵢ - γ̂ᴰₜ

2. **τ-heterogeneity splitting**: Splits maximize:
   - Δ(Sₗ, Sᵣ) = (nₗ·nᵣ/n²) · (τ̂ₗ - τ̂ᵣ)²

3. **IV-style leaf estimation**:
   - τ̂ = Σ D̃Ỹ / Σ D̃²

See [docs/methods.md](docs/methods.md) for full methodology.

## Citation

If you use this package in your research, please cite the original CFFE paper:

```bibtex
@article{kattenberg2023causal,
  title={Causal Forests with Fixed Effects for Treatment Effect Heterogeneity in Difference-in-Differences},
  author={Kattenberg, Mark A.C. and Scheer, Bas J. and Thiel, Jurre H.},
  journal={CPB Discussion Paper},
  year={2023},
  institution={Netherlands Institute for Economic Policy Analysis (CPB)}
}
```

And optionally this Python implementation:

```bibtex
@software{causalfe,
  title = {causalfe: Causal Forests with Fixed Effects in Python},
  author = {Aytug, Harry},
  year = {2026},
  url = {https://github.com/haytug/causalfe}
}
```

## References

- **Kattenberg, M.A.C., Scheer, B.J., & Thiel, J.H. (2023).** Causal Forests with Fixed Effects for Treatment Effect Heterogeneity in Difference-in-Differences. *CPB Discussion Paper*. — **The foundational paper for this implementation.**
- Athey, S., & Imbens, G. (2016). Recursive Partitioning for Heterogeneous Causal Effects. *PNAS*.
- Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *JASA*.

## License

MIT License - see [LICENSE](LICENSE) for details.
