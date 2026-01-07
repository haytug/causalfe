"""
Fixed effects residualization for CFFE.

Two-way FE via iterative demeaning (converges in ~3-5 iterations).
"""

import numpy as np


def fe_residualize(
    Y: np.ndarray,
    D: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    iters: int = 5,
) -> tuple:
    """
    Two-way fixed effects residualization via iterative demeaning.

    Parameters
    ----------
    Y : array
        Outcome variable.
    D : array
        Treatment variable.
    unit : array
        Unit identifiers.
    time : array
        Time identifiers.
    iters : int
        Number of demeaning iterations (default 5).

    Returns
    -------
    Y_tilde, D_tilde : tuple of arrays
        Residualized outcome and treatment.
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    unit = np.asarray(unit)
    time = np.asarray(time)

    Y_res = Y.copy()
    D_res = D.copy()

    # Get unique units and times
    unique_units = np.unique(unit)
    unique_times = np.unique(time)

    for _ in range(iters):
        # Unit FE demeaning
        for u in unique_units:
            mask = unit == u
            if mask.sum() > 0:
                Y_res[mask] -= Y_res[mask].mean()
                D_res[mask] -= D_res[mask].mean()

        # Time FE demeaning (on already unit-demeaned values)
        for t in unique_times:
            mask = time == t
            if mask.sum() > 0:
                Y_res[mask] -= Y_res[mask].mean()
                D_res[mask] -= D_res[mask].mean()

    return Y_res, D_res


def fe_residualize_single(
    Y: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    iters: int = 5,
) -> np.ndarray:
    """Residualize a single variable."""
    Y = np.asarray(Y, dtype=float)
    unit = np.asarray(unit)
    time = np.asarray(time)

    Y_res = Y.copy()
    unique_units = np.unique(unit)
    unique_times = np.unique(time)

    for _ in range(iters):
        for u in unique_units:
            mask = unit == u
            if mask.sum() > 0:
                Y_res[mask] -= Y_res[mask].mean()

        for t in unique_times:
            mask = time == t
            if mask.sum() > 0:
                Y_res[mask] -= Y_res[mask].mean()

    return Y_res
