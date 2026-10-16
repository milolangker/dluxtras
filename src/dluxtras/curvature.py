"""Unit-free (diagonally equilibrated) Hessians.

A Hessian in physical units has entries `d2L / dx_i dx_j`, so its rows and columns
carry each parameter's units. A Zernike coefficient in metres with a nm-scale effect
gives entries ~1e18 larger than a parameter of order one, and the matrix's condition
number then reflects units rather than real degeneracy between parameters.

These functions hold the Hessian as

    H = normalised / outer(s, s),    s = 1 / sqrt(|diag(H)|)

where `normalised` is dimensionless with a unit diagonal (Jacobi scaling) and `s`
carries the units. For a Fisher matrix, `s_i` is the conditional 1-sigma of `x_i` (all
other parameters fixed). Do linear algebra on `normalised` and reapply `s` at the end;
never invert the physical-units matrix directly, and never apply damping, ridges or
eigenvalue cutoffs to it.

Normalisation removes ill-conditioning caused by units only. Degeneracy between
parameters (correlations near +-1) survives it: `cond(normalised)` measures that, and
is within a factor n of the best any diagonal rescaling can achieve (van der Sluis,
1969). `dluxtras.fitting` builds on these functions.
"""

import jax
import jax.numpy as np
import jax.scipy.linalg as jsl
from jax.flatten_util import ravel_pytree
from zodiax.diffops import hessian

__all__ = ["scaled_hessian", "unscale", "scaled_covariance", "correlation"]


def _flat_scales(scales, x):
    """Flatten scales (None, scalar, or pytree matching x) like ravel_pytree(x)."""
    if scales is None:
        scales = 1.0
    if jax.tree_util.treedef_is_leaf(jax.tree_util.tree_structure(scales)):
        scales = jax.tree_util.tree_map(lambda leaf: scales, x)
    full = jax.tree_util.tree_map(
        lambda s, leaf: np.broadcast_to(np.asarray(s, np.result_type(leaf)), np.shape(leaf)),
        scales,
        x,
    )
    return ravel_pytree(full)[0]


def scaled_hessian(f, x, scales=None, *, nbatches=1, jit=True, checkpoint=False):
    """Hessian of a scalar function, returned in unit-free form.

    Differentiates `f(x0 + c * u)` at `u = 0` with `zodiax.diffops.hessian`, then
    normalises the result to a unit diagonal. Rescaling does not change the accuracy of
    the Hessian itself (floating point and autodiff are scale-invariant); `scales` only
    keeps intermediate values away from float over/underflow.

    Parameters
    ----------
    f : callable
        Scalar function of a pytree, e.g. a negative log likelihood.
    x : PyTree
        Point to evaluate at, e.g. `{"coefficients": ..., "position": ...}`.
    scales : None, float or PyTree, optional
        Rough characteristic size of each parameter in its own units: a scalar, or a
        pytree matching `x` whose leaves broadcast to `x`'s leaves, e.g.
        `{"coefficients": 1e-9, "position": 1e-7}`. Order-of-magnitude guesses are
        enough. Default `None` (all ones).
    nbatches : int, optional
        Passed to `zodiax.diffops.hessian` (memory). Default 1.
    jit : bool, optional
        Passed to `zodiax.diffops.hessian`. Default True.
    checkpoint : bool, optional
        Passed to `zodiax.diffops.hessian`. Default False.

    Returns
    -------
    normalised : Array
        `(n, n)` Hessian with unit diagonal: zero where a parameter has no curvature
        and -1 where its curvature is negative.
    s : Array
        `(n,)` scales in `x`'s units, with `H = normalised / outer(s, s)`.
    unflatten : callable
        Maps a flat `(n,)` vector back to `x`'s structure.

    Raises
    ------
    ValueError
        If `scales` are not finite and positive.
    """
    x_flat, unflatten = ravel_pytree(x)
    c = _flat_scales(scales, x)
    if not bool(np.all(np.isfinite(c) & (c > 0))):
        raise ValueError("scales must be finite and positive")

    # Differentiate in u where x = x0 + c * u, evaluated at u = 0.
    H_u, _ = hessian(
        lambda u: f(unflatten(x_flat + c * u)),
        np.zeros_like(x_flat),
        nbatches=nbatches,
        jit=jit,
        checkpoint=checkpoint,
    )
    H_u = (H_u + H_u.T) / 2

    # Jacobi-normalise. Parameters with zero curvature keep their input scale.
    d = np.abs(np.diag(H_u))
    r = np.where(d > 0, 1 / np.sqrt(np.where(d > 0, d, 1)), 1)
    normalised = r[:, None] * H_u * r[None, :]
    return normalised, c * r, unflatten


def unscale(normalised, s):
    """The Hessian in the parameters' own units, `normalised / outer(s, s)`.

    In float32 this can over/underflow when `s` spans many decades; prefer working with
    `(normalised, s)` directly.

    Parameters
    ----------
    normalised : Array
        `(n, n)` unit-free matrix.
    s : Array
        `(n,)` scales.

    Returns
    -------
    Array
        `(n, n)` matrix in original units.
    """
    return normalised / (s[:, None] * s[None, :])


def scaled_covariance(normalised, s):
    """Covariance `inv(H)` in the parameters' own units, via Cholesky of `normalised`.

    Requires `normalised` to be positive definite (e.g. a Fisher matrix, or the Hessian
    of a negative log likelihood at a minimum); otherwise the result is NaN.

    Parameters
    ----------
    normalised : Array
        `(n, n)` unit-free matrix.
    s : Array
        `(n,)` scales.

    Returns
    -------
    Array
        `(n, n)` covariance. Marginal uncertainties are `np.sqrt(np.diag(cov))`.
    """
    factor = jsl.cho_factor(normalised, lower=True)
    inverse = jsl.cho_solve(factor, np.eye(normalised.shape[0], dtype=normalised.dtype))
    inverse = (inverse + inverse.T) / 2
    return s[:, None] * inverse * s[None, :]


def correlation(normalised):
    """Correlation matrix of the covariance `inv(H)`, from `normalised` alone.

    The scales cancel, so `s` is not needed.

    Parameters
    ----------
    normalised : Array
        `(n, n)` unit-free, positive-definite matrix.

    Returns
    -------
    Array
        `(n, n)` correlation matrix with unit diagonal.
    """
    factor = jsl.cho_factor(normalised, lower=True)
    inverse = jsl.cho_solve(factor, np.eye(normalised.shape[0], dtype=normalised.dtype))
    d = 1 / np.sqrt(np.diag(inverse))
    return d[:, None] * inverse * d[None, :]
