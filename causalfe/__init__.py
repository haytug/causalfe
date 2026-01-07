"""
causalfe - Causal Forests with Fixed Effects

A Python implementation of causal forests for panel data with
unit and time fixed effects.

Main classes:
- CFFEForest: The main estimator

Inference functions:
- half_sample_variance: Fast variance estimation (default)
- jackknife_variance: More stable variance estimation
- cluster_robust_variance: Cluster-robust variance for ATE
- cluster_bootstrap_variance: Full cluster bootstrap
- confidence_interval: Construct CIs from variance estimates
"""

from .forest import CFFEForest
from .inference import (
    half_sample_variance,
    multi_split_variance,
    jackknife_variance,
    infinitesimal_jackknife_variance,
    cluster_robust_variance,
    cluster_robust_se,
    cluster_bootstrap_variance,
    confidence_interval,
)

__version__ = "0.1.0"

__all__ = [
    # Main estimator
    "CFFEForest",
    # Variance estimation
    "half_sample_variance",
    "multi_split_variance",
    "jackknife_variance",
    "infinitesimal_jackknife_variance",
    "cluster_robust_variance",
    "cluster_robust_se",
    "cluster_bootstrap_variance",
    # Confidence intervals
    "confidence_interval",
]
