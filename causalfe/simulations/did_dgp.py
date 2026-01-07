"""
Data Generating Processes for CFFE validation.

Key insight: CFFE requires within-unit AND within-time variation in D.
Standard DiD (everyone treated at same time) won't work because D is
collinear with time FE.

Solutions:
1. Staggered adoption (different units treated at different times)
2. Partial treatment (some units never treated)
3. Treatment intensity variation
"""

import numpy as np


def dgp_fe_only(N=100, T=5, seed=0):
    """
    FE-only DGP: no treatment effect (placebo).
    
    Uses staggered adoption structure but with τ=0.
    Used to verify τ̂ ≈ 0 (no spurious heterogeneity).
    """
    rng = np.random.default_rng(seed)
    n = N * T

    X = rng.normal(size=(n, 3))
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)

    alpha = rng.normal(size=N)
    gamma = rng.normal(size=T)

    # Staggered adoption (for identification) but NO treatment effect
    adoption_time = rng.integers(1, T, size=N)
    D = (time >= adoption_time[unit]).astype(float)
    
    # Y has NO treatment effect
    Y = alpha[unit] + gamma[time] + rng.normal(size=n)

    tau_true = np.zeros(n)
    return X, Y, D, unit, time, tau_true


def dgp_did_homogeneous(N=100, T=5, tau=2.0, seed=0):
    """
    Homogeneous DiD: constant treatment effect τ for all units.
    
    Uses staggered adoption for identification.
    """
    rng = np.random.default_rng(seed)
    n = N * T

    X = rng.normal(size=(n, 3))
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)

    alpha = rng.normal(size=N)
    gamma = rng.normal(size=T)

    # Staggered adoption
    adoption_time = rng.integers(1, T, size=N)
    D = (time >= adoption_time[unit]).astype(float)
    
    Y = tau * D + alpha[unit] + gamma[time] + rng.normal(size=n)

    tau_true = np.full(n, tau)
    return X, Y, D, unit, time, tau_true


def dgp_did_heterogeneous(N=100, T=5, seed=0):
    """
    Heterogeneous DiD: treatment effect varies with X.
    
    Uses staggered adoption for identification.
    τ(x) = X[:, 0] (heterogeneous by first covariate)
    """
    rng = np.random.default_rng(seed)
    n = N * T

    X = rng.normal(size=(n, 3))
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)

    # Heterogeneous effect based on X
    tau_true = X[:, 0]

    # Staggered adoption
    adoption_time = rng.integers(1, T, size=N)
    D = (time >= adoption_time[unit]).astype(float)

    alpha = rng.normal(size=N)
    gamma = rng.normal(size=T)

    Y = tau_true * D + alpha[unit] + gamma[time] + rng.normal(size=n)

    return X, Y, D, unit, time, tau_true


def dgp_staggered(N=100, T=5, seed=0):
    """
    Staggered adoption: units adopt treatment at different times.
    Treatment effect is heterogeneous (varies with X[:, 0]).
    
    This is the canonical CFFE setting.
    """
    rng = np.random.default_rng(seed)
    n = N * T

    X = rng.normal(size=(n, 3))
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)

    # Staggered adoption: each unit has a random adoption time
    adoption_time = rng.integers(1, T, size=N)
    D = (time >= adoption_time[unit]).astype(float)

    tau_true = X[:, 0]  # Heterogeneous effect

    alpha = rng.normal(size=N)
    gamma = rng.normal(size=T)

    Y = tau_true * D + alpha[unit] + gamma[time] + rng.normal(size=n)

    return X, Y, D, unit, time, tau_true


def dgp_partial_treatment(N=100, T=5, treat_frac=0.5, seed=0):
    """
    Partial treatment: only some units ever get treated.
    
    This creates within-time variation even with simultaneous adoption.
    """
    rng = np.random.default_rng(seed)
    n = N * T

    X = rng.normal(size=(n, 3))
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)

    # Only some units are ever treated
    ever_treated = rng.random(N) < treat_frac
    D = ((time >= T // 2) & ever_treated[unit]).astype(float)

    tau_true = X[:, 0]

    alpha = rng.normal(size=N)
    gamma = rng.normal(size=T)

    Y = tau_true * D + alpha[unit] + gamma[time] + rng.normal(size=n)

    return X, Y, D, unit, time, tau_true
