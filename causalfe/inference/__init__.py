"""
Inference module for CFFE.

Provides variance estimation and confidence intervals that are valid
for panel data with clustered observations.

Methods:
- half_sample_variance: Fast, default method (GRF-style)
- jackknife_variance: More stable, slower
- cluster_bootstrap_variance: Most robust, slowest
- cluster_robust_variance: For ATE inference
"""

from .half_sample import (
    half_sample_variance,
    multi_split_variance,
    jackknife_variance,
    infinitesimal_jackknife_variance,
)
from .cluster_robust import (
    cluster_robust_variance,
    cluster_robust_se,
    cluster_bootstrap_variance,
    wild_cluster_bootstrap_variance,
    confidence_interval,
)

__all__ = [
    # Half-sample methods
    "half_sample_variance",
    "multi_split_variance",
    "jackknife_variance",
    "infinitesimal_jackknife_variance",
    # Cluster-robust methods
    "cluster_robust_variance",
    "cluster_robust_se",
    "cluster_bootstrap_variance",
    "wild_cluster_bootstrap_variance",
    # Confidence intervals
    "confidence_interval",
]
