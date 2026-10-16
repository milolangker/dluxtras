"""Shared machinery for Fisher eigenmodes with marginalised nuisance parameters.

`dluxtras.spectral_eigenmodes` and `dluxtras.wavefront_eigenmodes` both compute the
Fisher matrix of one parameter vector `q` of a model (bin fluxes, or per-pixel OPD),
optionally marginalised over nuisance parameters, and its leading eigenmodes. This
module holds the parts they share.

Nuisance parameters are given as a list whose entries can be mixed freely:

- a model key (str), e.g. `"source.position"` or `"optics.psf_pixel_scale"`: every
  entry of that parameter becomes a nuisance;
- an array of directions in `q`-space, shape `(len(q),)` or `(len(q), m)`, e.g.
  `wavefront_eigenmodes.nuisance_shapes(layer)` for tip, tilt and defocus: the
  coefficients of those directions become nuisances.

`marginalise` nuisances are per model: every model (e.g. every exposure) gets its own
independent copy, as for per-exposure source positions. `marginalise_shared`
nuisances are one set shared by all models, as for a common pixel scale. Both are
marginalised together, exactly, via the Schur complement of the joint Fisher matrix:

    F_marg = F_qq - F_qn F_nn^+ F_nq

i.e. the information about `q` left once all nuisances are fitted freely. `F_nn` is
inverted in Jacobi-normalised (unit-free) form, so nuisances with very different
units (radians, arcseconds, photons) can be mixed, and a pseudo-inverse drops nuisance
combinations the data cannot see at all (e.g. OPD piston).

All Fisher matrices here are `sum J.T W J` with per-pixel weights `W = 1 / variance`
frozen at the noiseless reference prediction (pixels uncorrelated). Enable
`jax_enable_x64` before building models.
"""

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as np
import numpy as onp
from jax.flatten_util import ravel_pytree
from scipy.sparse.linalg import LinearOperator, eigsh

__all__ = ["FisherOperator", "Eigenpairs", "leading_modes", "noise_weights"]


def noise_weights(mu, *, noise="poisson", variance=1.0, variance_fn=None, bad_mask=None):
    """Per-pixel Fisher weights `1 / variance`, zero at excluded pixels.

    Parameters
    ----------
    mu : Array
        Noiseless reference prediction (any shape, e.g. an image or a stack).
    noise : {"poisson", "gaussian"}, optional
        `"poisson"` (default) uses `variance = mu`, so `mu` must be in counts and
        strictly positive on included pixels. `"gaussian"` uses `variance`.
    variance : float or Array, optional
        Gaussian variance (not standard deviation): a scalar or an array broadcastable to
        `mu`. Ignored for Poisson noise or when `variance_fn` is given. Default 1.
    variance_fn : callable, optional
        `variance_fn(mu) -> variance` overrides both noise choices, e.g.
        `lambda mu: mu + read_noise**2`. Evaluated once, at the reference.
    bad_mask : Array, optional
        Boolean, the shape of `mu`; True excludes a pixel.

    Returns
    -------
    Array
        Weights with the shape of `mu`.

    Raises
    ------
    ValueError
        If `noise` is invalid, the mask does not match `mu` or excludes every pixel, the
        reference is not finite (or negative, for Poisson) on included pixels, or the
        included variances are not finite and positive.
    """
    if noise not in ("gaussian", "poisson"):
        raise ValueError("noise must be 'gaussian' or 'poisson'")
    mu = onp.asarray(mu, dtype=float)
    keep = onp.ones(mu.shape, dtype=bool) if bad_mask is None else ~onp.asarray(bad_mask, bool)
    if keep.shape != mu.shape or not keep.any():
        raise ValueError("Bad mask must match the model output and retain pixels")
    if not onp.isfinite(mu[keep]).all() or (noise == "poisson" and onp.any(mu[keep] < 0)):
        raise ValueError("Invalid reference prediction on included pixels")
    if variance_fn is not None:
        var = variance_fn(np.asarray(mu))
    else:
        var = mu if noise == "poisson" else variance
    var = onp.broadcast_to(onp.asarray(var, dtype=float), mu.shape)
    if not onp.all(onp.isfinite(var[keep]) & (var[keep] > 0)):
        raise ValueError("Included variances must be finite and positive; mask zero-mean "
                         "Poisson pixels explicitly")
    return np.asarray(onp.where(keep, 1 / onp.where(keep, var, 1.0), 0.0))


def _each(value, n_models, name):
    """A list/tuple gives one entry per model; anything else is used for every model."""
    if isinstance(value, (list, tuple)):
        if len(value) != n_models:
            raise ValueError(f"Supply one entry of {name} per model")
        return list(value)
    return [value] * n_models


@dataclass
class _Linearisation:
    """One model's prediction, linearised in q (flattened outputs)."""

    model: Any
    weight: Any  # flat per-pixel weights, zero where excluded
    pushforward: Callable  # v (n_q,) -> J v (n_pix,)
    pullback: Callable  # t (n_pix,) -> J.T t (n_q,)


def _split(entries, n_q):
    """Separate nuisance entries into model keys and (n_q, m) direction columns."""
    entries = [] if entries is None else list(entries)
    keys = [e for e in entries if isinstance(e, str)]
    directions = []
    for entry in entries:
        if isinstance(entry, str):
            continue
        d = onp.asarray(entry, dtype=float)
        if d.ndim == 1:
            d = d[:, None]
        if d.ndim != 2 or d.shape[0] != n_q:
            raise ValueError(f"Nuisance directions must have shape ({n_q},) or ({n_q}, m)")
        directions.append(d)
    return keys, (onp.concatenate(directions, axis=1) if directions else onp.zeros((n_q, 0)))


def _nuisance_columns(lin, keys, directions):
    """(n_pix, m) derivatives of one model's prediction w.r.t. its nuisances."""
    columns = []
    if keys:
        values = {key: np.asarray(lin.model.get(key), dtype=float) for key in keys}
        flat, unflatten = ravel_pytree(values)

        def predict(eta):
            new = unflatten(eta)
            return np.asarray(lin.model.set(list(new), list(new.values())).model()).ravel()

        columns.append(jax.jacfwd(predict)(flat))
    for j in range(directions.shape[1]):
        columns.append(lin.pushforward(np.asarray(directions[:, j]))[:, None])
    if not columns:
        return np.zeros((lin.weight.size, 0))
    return np.concatenate(columns, axis=1)


def _schur_terms(lins, n_q, marginalise, marginalise_shared, rcond):
    """G = F_qn and C^+ = pinv(F_nn) for the joint nuisance vector, or None if none."""
    keys, directions = _split(marginalise, n_q)
    shared_keys, shared_directions = _split(marginalise_shared, n_q)
    per = [_nuisance_columns(lin, keys, directions) for lin in lins]
    shared = [_nuisance_columns(lin, shared_keys, shared_directions) for lin in lins]
    m_per, m_shared = per[0].shape[1], shared[0].shape[1]
    n_eta = len(lins) * m_per + m_shared
    if n_eta == 0:
        return None
    G = onp.zeros((n_q, n_eta))
    C = onp.zeros((n_eta, n_eta))
    s0 = len(lins) * m_per  # shared block starts here
    for i, (lin, Bp, Bs) in enumerate(zip(lins, per, shared)):
        w = lin.weight[:, None]
        p0 = i * m_per
        for block, start in ((Bp, p0), (Bs, s0)):
            for j in range(block.shape[1]):
                G[:, start + j] += onp.asarray(lin.pullback(w[:, 0] * block[:, j]))
        C[p0:p0 + m_per, p0:p0 + m_per] += onp.asarray(Bp.T @ (w * Bp))
        C[p0:p0 + m_per, s0:] += onp.asarray(Bp.T @ (w * Bs))
        C[s0:, p0:p0 + m_per] += onp.asarray(Bs.T @ (w * Bp))
        C[s0:, s0:] += onp.asarray(Bs.T @ (w * Bs))
    C = (C + C.T) / 2
    # Pseudo-inverse in unit-free (Jacobi-normalised) form: mixed units are safe, and
    # nuisance combinations with no effect on the data are dropped.
    d = onp.diag(C)
    s = onp.where(d > 0, 1 / onp.sqrt(onp.where(d > 0, d, 1.0)), 0.0)
    C_pinv = s[:, None] * onp.linalg.pinv(s[:, None] * C * s[None, :], rcond=rcond,
                                          hermitian=True) * s[None, :]
    return G, C_pinv


@dataclass
class FisherOperator:
    """Matrix-free Fisher matrix: `F(v)` returns the product `F @ v`.

    Attributes
    ----------
    size : int
        Length of the parameter vector `q`.
    matvec : callable
        `matvec(v) -> F @ v` for a vector `v` of length `size`.
    """

    size: int
    matvec: Callable

    def __call__(self, v):
        """Return `F @ v` as a NumPy array."""
        return onp.asarray(self.matvec(onp.asarray(v, dtype=float).reshape(self.size)))

    def dense(self):
        """The full `(size, size)` matrix, built column by column (small `size` only)."""
        columns = [self(e) for e in onp.eye(self.size)]
        F = onp.column_stack(columns)
        return (F + F.T) / 2


def _assemble(lins, n_q, marginalise, marginalise_shared, rcond):
    """FisherOperator for sum_m J_m.T W_m J_m, with nuisances marginalised."""
    products = [jax.jit(lambda v, lin=lin: lin.pullback(lin.weight * lin.pushforward(v)))
                for lin in lins]
    terms = _schur_terms(lins, n_q, marginalise, marginalise_shared, rcond)

    def matvec(v):
        v = np.asarray(v, dtype=float)
        out = sum(onp.asarray(product(v)) for product in products)
        if terms is not None:
            G, C_pinv = terms
            out = out - G @ (C_pinv @ (G.T @ onp.asarray(v)))
        return out

    return FisherOperator(n_q, matvec)


class Eigenpairs(NamedTuple):
    """Leading eigenpairs of a Fisher matrix.

    Attributes
    ----------
    values : Array
        Eigenvalues, descending.
    vectors : Array
        Unit-norm eigenvectors as columns, shape `(n, k)`; the sign makes each column's
        largest-magnitude entry positive.
    residuals : Array
        Relative residuals `|F v - lambda v| / |lambda|`, to check convergence.
    """

    values: Any
    vectors: Any
    residuals: Any


def leading_modes(fisher, k=6, *, project_out=None, tol=1e-7, ncv=None, maxiter=None, seed=0):
    """The `k` largest eigenvalues and eigenvectors of a Fisher matrix.

    A dense matrix is diagonalised exactly (`eigh`); a `FisherOperator` (or any
    `matvec` with a `size`) uses restarted Lanczos (SciPy `eigsh`), which never forms
    the matrix.

    Parameters
    ----------
    fisher : Array or FisherOperator
        Symmetric positive semi-definite Fisher matrix, dense `(n, n)` or matrix-free.
    k : int, optional
        Number of modes. Default 6.
    project_out : Array, optional
        Directions `(n,)` or `(n, m)` to project out before diagonalising, e.g. a column
        of ones for OPD piston. The modes are then orthogonal to them. Default None.
    tol : float, optional
        Lanczos convergence tolerance (matrix-free only). Default 1e-7.
    ncv : int, optional
        Lanczos vectors (matrix-free only). Default `max(2k + 1, 20)`, capped at `n`.
    maxiter : int, optional
        Lanczos iterations (matrix-free only). Default SciPy's.
    seed : int, optional
        Seed of the Lanczos start vector (matrix-free only). Default 0.

    Returns
    -------
    Eigenpairs
        `(values, vectors, residuals)`. Dense eigenvalues below roundoff are set to 0.

    Raises
    ------
    ValueError
        If `k` is out of range, the matrix is not square, or (dense) it has a
        significantly negative eigenvalue.
    FloatingPointError
        If a matrix-free product is not finite.
    """
    dense = not isinstance(fisher, FisherOperator) and not callable(fisher)
    if dense:
        fisher = onp.asarray(fisher, dtype=float)
        if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
            raise ValueError("fisher must be square")
        n = fisher.shape[0]
    else:
        n = int(fisher.size)
    Q = None
    if project_out is not None:
        P = onp.asarray(project_out, dtype=float).reshape(n, -1)
        Q = onp.linalg.qr(P)[0]
    available = n - (0 if Q is None else Q.shape[1])
    if not isinstance(k, (int, onp.integer)) or not 1 <= k <= available - (0 if dense else 1):
        raise ValueError(f"Require 1 <= k <= {available - (0 if dense else 1)} for this matrix")

    def project(v):
        return v if Q is None else v - Q @ (Q.T @ v)

    if dense:
        F = (fisher + fisher.T) / 2
        if Q is not None:
            F = project(project(F).T).T
        values, vectors = onp.linalg.eigh(F)
        values, vectors = values[::-1], vectors[:, ::-1]
        scale = max(float(onp.max(onp.abs(values))), onp.finfo(float).tiny)
        if values[-1] < -1e-9 * scale:
            raise ValueError("Significantly negative Fisher eigenvalue")
        values = onp.where(values > 10 * n * onp.finfo(float).eps * scale, values, 0.0)
        values, vectors = values[:k], vectors[:, :k]

        def apply(v):
            return F @ v
    else:
        def apply(v):
            v = project(onp.asarray(v, dtype=float).reshape(n))
            result = onp.asarray(fisher(v), dtype=float).reshape(n)
            if not onp.isfinite(result).all():
                raise FloatingPointError("Non-finite Fisher-vector product")
            return project(result)

        operator = LinearOperator((n, n), matvec=apply, dtype=onp.float64)
        ncv = min(n, max(2 * k + 1, 20)) if ncv is None else ncv
        v0 = project(onp.random.default_rng(seed).normal(size=n))
        values, vectors = eigsh(operator, k=k, which="LA", v0=v0, tol=tol, ncv=ncv,
                                maxiter=maxiter)
        order = onp.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
    # Consistent display sign; eigenvector signs have no physical meaning.
    pivots = onp.argmax(onp.abs(vectors), axis=0)
    vectors = vectors * onp.where(vectors[pivots, onp.arange(k)] < 0, -1.0, 1.0)
    denominator = onp.maximum(onp.abs(values), onp.finfo(float).tiny)
    residuals = onp.array([onp.linalg.norm(apply(vectors[:, i]) - values[i] * vectors[:, i])
                           / denominator[i] for i in range(k)])
    return Eigenpairs(values, vectors, residuals)
