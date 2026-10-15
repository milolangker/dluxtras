"""Unit-free (diagonally equilibrated) Hessians.

A Hessian in physical units has entries d2L/dx_i dx_j, so its rows and columns
carry the units of each parameter: a coefficient in metres with a nm-scale
effect gives entries ~1e18 larger than one with unit scale. Its condition
number is then dominated by units rather than by real parameter degeneracy.

These functions reparameterise x = x0 + scales * u, differentiate in u, and
then symmetrically normalise the result to unit diagonal (Jacobi scaling):

    H = normalised / outer(s, s),    s = 1 / sqrt(|diag(H)|)

normalised is dimensionless and ~O(1); s carries the units. For a Fisher
matrix, s_i is the conditional 1-sigma uncertainty of x_i (all others fixed)
and normalised is the Fisher matrix of the standardised parameters.
Do linear algebra (solves, inverses, eigendecompositions) on normalised and
only reapply s at the end; never invert the physical-units matrix directly.

Normalisation removes ill-conditioning caused by units only. Degeneracy
between parameters (correlations near +-1) survives it: cond(normalised)
measures that, and is within a factor n of the best any diagonal rescaling
can achieve (van der Sluis, 1969).
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
    """Hessian of scalar f at pytree x, returned in unit-free form.

    scales: rough characteristic size of each parameter in its own units,
    as None (all ones), a scalar, or a pytree matching x whose leaves
    broadcast to x's leaves, e.g. {'coefficients': 1e-9, 'position': 1e-7}.
    Order-of-magnitude guesses are enough: they only keep intermediate
    values far from float over/underflow during differentiation. The
    exact normalisation is computed from the result.

    nbatches, jit and checkpoint are passed to zodiax.diffops.hessian.

    Returns (normalised, s, unflatten): normalised is (n, n) with unit
    diagonal (zero where a parameter has no curvature, -1 where it is
    negative), s is (n,) in x's units with H = normalised / outer(s, s),
    and unflatten maps a flat (n,) vector back to x's structure.
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
    """The Hessian in original units, normalised / outer(s, s).

    In float32 this can over/underflow when s spans many decades; prefer
    working with (normalised, s) directly.
    """
    return normalised / (s[:, None] * s[None, :])


def scaled_covariance(normalised, s):
    """Covariance inv(H) in original units, via Cholesky of normalised.

    Requires normalised to be positive definite (e.g. a Fisher matrix, or
    the Hessian of a negative log likelihood at a minimum); otherwise the
    result is NaN. Marginal uncertainties are np.sqrt(np.diag(cov)).
    """
    factor = jsl.cho_factor(normalised, lower=True)
    inverse = jsl.cho_solve(factor, np.eye(normalised.shape[0], dtype=normalised.dtype))
    inverse = (inverse + inverse.T) / 2
    return s[:, None] * inverse * s[None, :]


def correlation(normalised):
    """Parameter correlation matrix (of the covariance) from normalised.

    Unit-free, so it never needs s: the scales cancel.
    """
    factor = jsl.cho_factor(normalised, lower=True)
    inverse = jsl.cho_solve(factor, np.eye(normalised.shape[0], dtype=normalised.dtype))
    d = 1 / np.sqrt(np.diag(inverse))
    return d[:, None] * inverse * d[None, :]
