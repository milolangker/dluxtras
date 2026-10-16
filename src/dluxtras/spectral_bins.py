"""Top-hat wavelength bins that best match a set of spectral eigenmodes.

Fitting a spectrum in its Fisher eigenmode basis constrains only the first `K` modes;
the rest are set by the reference spectrum, not by the data. To report the recovered
spectrum as `K` top-hat bins (photons, or flux density, per bin), choose bins whose
averaging functionals lie inside the span of those `K` modes: then each bin's value is
fixed by the fitted modes alone, and the unconstrained modes cannot leak into it.

`optimal_bins` places the bin edges to minimise that leakage. With `B` the bin matrix
(row `k` is the fraction of each fine wavelength cell inside bin `k`; see `bin_matrix`)
and `P = V V.T` the projector onto the retained modes `V` (orthonormal columns), the
leakage of bin `k` is

    L_k = |B_k (I - P)|^2 / |B_k|^2  =  1 - |B_k V|^2 / |B_k|^2,

the fraction of the bin outside the modes' span, and the objective is the mean of `L_k`
over bins. Every mode counts equally (`P` is unweighted). Because top-hats do not
overlap, the normalised bins are orthonormal, so for `K` bins and `K` modes this also
equals how badly the bins represent the modes, `|V - B_n B_n.T V|^2 / K`: both readings of
"projection error" have the same optimum.

For equal-width bins this is the same objective as
`sum_k |B_k (I - P)|^2 / sum_k |B_k|^2` (all rows have the same norm, up to cells cut by
an edge); with free widths, normalising per bin stops the optimiser from hiding a poorly
represented bin by making it narrow.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as np
import numpy as onp
from scipy.optimize import minimize

__all__ = ["optimal_bins", "bin_matrix", "native_edges", "SpectralBins"]


def native_edges(wavelengths):
    """Edges of the fine wavelength cells, from their centres.

    Interior edges are midpoints between neighbouring centres; the outer edges extend
    by half the neighbouring spacing.

    Parameters
    ----------
    wavelengths : Array
        Increasing cell centres, shape (n,), e.g. a spectrum's `wavelengths`.

    Returns
    -------
    Array
        Edges, shape (n + 1,).
    """
    w = onp.asarray(wavelengths, dtype=float)
    if w.ndim != 1 or w.size < 2 or onp.any(onp.diff(w) <= 0):
        raise ValueError("wavelengths must be 1D, increasing, with at least two entries")
    mid = 0.5 * (w[:-1] + w[1:])
    return onp.concatenate([[w[0] - (mid[0] - w[0])], mid, [w[-1] + (w[-1] - mid[-1])]])


def bin_matrix(edges, cell_edges):
    """Fraction of each fine cell inside each bin, shape (n_bins, n_cells).

    `bin_matrix(edges, cell_edges) @ q` gives the photons in each bin for photons-per-cell
    `q`; divide by the bin widths for photons per unit wavelength. Differentiable in
    `edges`.

    Parameters
    ----------
    edges : Array
        Increasing bin edges, shape (n_bins + 1,), same units as `cell_edges`.
    cell_edges : Array
        Fine cell edges, shape (n_cells + 1,); see `native_edges`.

    Returns
    -------
    Array
        Overlap fractions, shape (n_bins, n_cells).
    """
    edges = np.asarray(edges)
    cell_edges = np.asarray(cell_edges)
    left = np.maximum(edges[:-1, None], cell_edges[None, :-1])
    right = np.minimum(edges[1:, None], cell_edges[None, 1:])
    return np.maximum(0.0, right - left) / np.diff(cell_edges)[None, :]


def _leakage(B, V):
    """Per-bin fraction outside span(V), 1 - |B_k V|^2 / |B_k|^2 (1 for an empty bin)."""
    norms = np.sum(B**2, axis=1)
    inside = np.sum((B @ V) ** 2, axis=1)
    return 1.0 - np.where(norms > 0, inside / np.where(norms > 0, norms, 1.0), 0.0)


@dataclass
class SpectralBins:
    """Result of `optimal_bins`: top-hat bins and how well they match the modes.

    Attributes
    ----------
    edges : Array
        Bin edges, shape (n_bins + 1,), in the units of the wavelengths (metres).
    matrix : Array
        `bin_matrix(edges, ...)`, shape (n_bins, n_cells): `matrix @ q` gives photons per
        bin for photons per fine cell `q`.
    leakage : Array
        Fraction of each bin outside the span of the modes, shape (n_bins,); 0 means the
        bin is fully determined by the fitted modes.
    objective : float
        Mean leakage, the minimised quantity.
    vectors : Array
        The modes the bins were optimised for, shape (n_cells, K).
    """

    edges: Any
    matrix: Any
    leakage: Any
    objective: float
    vectors: Any

    @property
    def centres(self):
        """Bin centres, shape (n_bins,)."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self):
        """Bin widths, shape (n_bins,)."""
        return onp.diff(self.edges)

    def bin(self, spectrum, covariance=None, *, density=False):
        """Bin a spectrum given in fine cells, propagating its covariance.

        Parameters
        ----------
        spectrum : Array
            Photons per fine cell, shape (n_cells,), e.g. `reference + V @ c` from a fit.
        covariance : Array, optional
            Covariance of `spectrum`, shape (n_cells, n_cells).
        density : bool, optional
            Divide by the bin widths, giving photons per unit wavelength. Default False.

        Returns
        -------
        values : Array
            Binned spectrum, shape (n_bins,).
        covariance : Array
            Its covariance, shape (n_bins, n_bins); only if `covariance` is given.
        """
        A = onp.asarray(self.matrix)
        if density:
            A = A / self.widths[:, None]
        values = A @ onp.asarray(spectrum)
        if covariance is None:
            return values
        return values, A @ onp.asarray(covariance) @ A.T

    def bin_modes(self, coefficients, coefficient_covariance, reference=None, *,
                  density=False):
        """Bin a spectrum fitted in the mode basis: `reference + vectors @ coefficients`.

        Parameters
        ----------
        coefficients : Array
            Fitted mode coefficients, shape (K,).
        coefficient_covariance : Array
            Their covariance, shape (K, K), e.g. the inverse Fisher matrix in the mode
            basis (`diag(1 / values)` for eigenmodes).
        reference : Array, optional
            Spectrum the modes perturb, shape (n_cells,). Default zero.
        density : bool, optional
            Divide by the bin widths. Default False.

        Returns
        -------
        values : Array
            Binned spectrum, shape (n_bins,).
        covariance : Array
            `(A V) Cov(c) (A V).T`, shape (n_bins, n_bins).
        """
        V = onp.asarray(self.vectors)
        spectrum = V @ onp.asarray(coefficients)
        if reference is not None:
            spectrum = spectrum + onp.asarray(reference)
        A = onp.asarray(self.matrix)
        if density:
            A = A / self.widths[:, None]
        AV = A @ V
        return A @ spectrum, AV @ onp.asarray(coefficient_covariance) @ AV.T


def optimal_bins(modes, wavelengths=None, n_bins=None, *, equal_widths=False,
                 max_edge_shift=0.1, n_starts=8, grid=64, seed=0):
    """Top-hat bin edges that best match the span of `K` spectral modes.

    Minimises the mean leakage, the average over bins of the fraction of each bin outside
    the modes' span (see the module docstring). The outer edges may move inward by up to
    `max_edge_shift` of the full span, to trim poorly constrained band tails; with
    `equal_widths=True` only those two outer edges are optimised and the bins between them
    have equal widths.

    The objective is bumpy in the edges (it has kinks wherever an edge crosses a fine cell
    boundary), so the search is global first: the outer edges are scanned on a `grid` x
    `grid` grid of equal-width solutions, and the best few are polished (Powell). With
    free widths, that equal-width optimum seeds a gradient search (L-BFGS) alongside
    equal-information and random starts, so free widths never do worse than equal ones.

    Parameters
    ----------
    modes : SpectralModes or Array
        The retained modes: a `SpectralModes` (its `vectors` and `wavelengths` are used),
        or an array of orthonormal columns, shape (n_cells, K).
    wavelengths : Array, optional
        Fine cell centres, shape (n_cells,). Required when `modes` is an array.
    n_bins : int, optional
        Number of bins. Default `K`, the number of modes. Use the number of constrainable
        modes, e.g. pass `modes.vectors[:, :K]` with `K = constrainable_modes(modes)`.
    equal_widths : bool, optional
        Keep the bins equal in width and optimise only the outer edges. Default False.
        Free bins are always at least one fine cell wide.
    max_edge_shift : float, optional
        Largest inward shift of each outer edge, as a fraction of the full span. 0 fixes
        the outer edges at the ends of the fine grid. Default 0.1.
    n_starts : int, optional
        Number of random free-width starts, in addition to the equal-width optimum and
        equal-information widths (bins containing equal shares of the projector's trace).
        Default 8.
    grid : int, optional
        Grid size per outer edge for the equal-width scan. Default 64.
    seed : int, optional
        Seed of the random starts. Default 0.

    Returns
    -------
    SpectralBins

    Raises
    ------
    ValueError
        If the modes are not orthonormal columns, the wavelengths do not match, or the
        settings are out of range.
    """
    if hasattr(modes, "vectors"):
        V = onp.asarray(modes.vectors, dtype=float)
        wavelengths = modes.wavelengths if wavelengths is None else wavelengths
    else:
        V = onp.asarray(modes, dtype=float)
    if wavelengths is None:
        raise ValueError("Pass wavelengths when modes is an array")
    if V.ndim != 2 or not onp.allclose(V.T @ V, onp.eye(V.shape[1]), atol=1e-8):
        raise ValueError("modes must be orthonormal columns, shape (n_cells, K)")
    cell_edges = native_edges(wavelengths)
    if cell_edges.size != V.shape[0] + 1:
        raise ValueError("wavelengths must have one entry per row of the modes")
    n_bins = V.shape[1] if n_bins is None else int(n_bins)
    if not 1 <= n_bins <= V.shape[0]:
        raise ValueError("n_bins must be between 1 and the number of fine cells")
    if not 0 <= max_edge_shift < 0.5:
        raise ValueError("max_edge_shift must be in [0, 0.5)")

    lo0, hi0 = float(cell_edges[0]), float(cell_edges[-1])
    shift = max_edge_shift * (hi0 - lo0)
    Vj, cell_edges_j = np.asarray(V), np.asarray(cell_edges)
    n_free = 0 if equal_widths else n_bins

    min_width = float(onp.min(onp.diff(cell_edges)))  # bins at least one fine cell wide

    def edges_from(params):
        # Outer edges by sigmoid within their allowed ranges; widths by softmax, with a
        # floor of one fine cell so that no bin can collapse.
        lo = lo0 + shift * jax.nn.sigmoid(params[0])
        hi = hi0 - shift * jax.nn.sigmoid(params[1])
        floor = np.minimum(min_width / (hi - lo), 0.5 / n_bins)
        fractions = (np.full(n_bins, 1.0 / n_bins) if equal_widths
                     else floor + (1 - n_bins * floor) * jax.nn.softmax(params[2:]))
        return lo + (hi - lo) * np.concatenate([np.zeros(1), np.cumsum(fractions)])

    @jax.jit
    def objective(params):
        return np.mean(_leakage(bin_matrix(edges_from(params), cell_edges_j), Vj))

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def fun(p):
        v, g = value_and_grad(np.asarray(p))
        return float(v), onp.asarray(g, dtype=float)

    def polish(x0, f):
        return minimize(lambda p: float(f(np.asarray(p))), x0, method="Powell",
                        options={"xtol": 1e-10, "ftol": 1e-14, "maxiter": 20000})

    # 1. Equal widths: scan the outer edges on a grid, then polish the best few.
    def equal_objective(ab):
        lo = lo0 + shift * jax.nn.sigmoid(ab[0])
        hi = hi0 - shift * jax.nn.sigmoid(ab[1])
        e = np.linspace(lo, hi, n_bins + 1)
        return np.mean(_leakage(bin_matrix(e, cell_edges_j), Vj))

    equal_objective = jax.jit(equal_objective)
    if shift > 0:
        # Grid uniform in edge position: sigmoid parameters of evenly spaced fractions.
        f = onp.linspace(0.005, 0.995, grid)
        g = onp.log(f / (1 - f))
        A, Bg = onp.meshgrid(g, g, indexing="ij")
        pairs = np.asarray(onp.stack([A.ravel(), Bg.ravel()], axis=1))
        values = onp.asarray(jax.vmap(equal_objective)(pairs))
        candidates = [onp.asarray(pairs[i]) for i in onp.argsort(values)[:4]]
        ab = min((polish(c, equal_objective) for c in candidates), key=lambda r: r.fun).x
    else:
        ab = onp.zeros(2)  # outer edges fixed at the ends of the grid
    best_x = onp.concatenate([ab, onp.zeros(n_free)])
    best_fun = float(equal_objective(np.asarray(ab)))

    # 2. Free widths: gradient search from the equal-width optimum and other starts.
    if not equal_widths:
        info = onp.cumsum(onp.sum(V**2, axis=1))  # cumulative trace of P over cells
        quantiles = onp.interp(onp.linspace(0, info[-1], n_bins + 1), onp.r_[0.0, info],
                               onp.arange(V.shape[0] + 1))
        info_fractions = onp.maximum(onp.diff(quantiles), 1e-3)
        rng = onp.random.default_rng(seed)
        width_starts = [onp.zeros(n_bins), onp.log(info_fractions / info_fractions.sum())]
        width_starts += [rng.normal(0, 0.5, n_bins) for _ in range(n_starts)]
        results = [minimize(fun, onp.concatenate([ab, w]), jac=True, method="L-BFGS-B")
                   for w in width_starts]
        results.sort(key=lambda r: r.fun)
        for r in results[:3]:  # derivative-free polish of the best few (kinks)
            r = min((r, polish(r.x, objective)), key=lambda s: s.fun)
            if r.fun < best_fun:
                best_x, best_fun = r.x, r.fun
    edges = onp.asarray(edges_from(np.asarray(best_x)))
    B = onp.asarray(bin_matrix(edges, cell_edges))
    leakage = onp.asarray(_leakage(np.asarray(B), Vj))
    return SpectralBins(edges, B, leakage, float(leakage.mean()), V)
