"""Optimal top-hat bins for spectral eigenmodes."""

import numpy as onp
import pytest

from dluxtras.spectral_bins import bin_matrix, native_edges, optimal_bins

WAVELENGTHS = onp.linspace(500e-9, 700e-9, 60)


def smooth_modes(k, n_cells=60):
    """Orthonormal smooth (polynomial) columns, like spectral eigenmodes."""
    x = onp.linspace(-1, 1, n_cells)
    return onp.linalg.qr(onp.vander(x, k, increasing=True))[0]


def tophat_modes(cell_ranges, n_cells=60):
    """Orthonormal columns: normalised top-hats over the given cell index ranges."""
    V = onp.zeros((n_cells, len(cell_ranges)))
    for k, (a, b) in enumerate(cell_ranges):
        V[a:b, k] = 1 / onp.sqrt(b - a)
    return V


def test_bin_matrix_and_native_edges():
    edges = native_edges(WAVELENGTHS)
    onp.testing.assert_allclose(onp.diff(edges), onp.diff(WAVELENGTHS)[0], rtol=1e-10)
    B = onp.asarray(bin_matrix(onp.linspace(edges[0], edges[-1], 4), edges))
    assert B.shape == (3, 60)
    onp.testing.assert_allclose(B.sum(axis=0), 1.0)  # every cell is fully assigned
    onp.testing.assert_allclose(B.sum(axis=1), 20.0)  # 20 cells per bin


def test_recovers_bins_that_the_modes_span():
    # Modes that are exactly three unequal top-hats: optimal bins reproduce them.
    ranges = [(0, 12), (12, 37), (37, 60)]
    bins = optimal_bins(tophat_modes(ranges), WAVELENGTHS, max_edge_shift=0)
    cell_edges = native_edges(WAVELENGTHS)
    onp.testing.assert_allclose(bins.edges, cell_edges[[0, 12, 37, 60]], atol=1e-3 * 3.4e-9)
    assert bins.objective < 1e-6 and onp.all(bins.leakage < 1e-6)

    # Equal-width top-hats inside a trimmed band: equal_widths finds the outer edges.
    equal = optimal_bins(tophat_modes([(6, 22), (22, 38), (38, 54)]), WAVELENGTHS,
                         equal_widths=True, max_edge_shift=0.2)
    onp.testing.assert_allclose(equal.edges, cell_edges[[6, 22, 38, 54]], atol=0.02 * 3.4e-9)
    assert equal.objective < 1e-4


def test_matches_width_weighted_objective_for_equal_widths():
    # For equal widths, the mean per-bin leakage equals
    # sum_k |B_k (I - P)|^2 / sum_k |B_k|^2, up to cells cut by an edge.
    V = smooth_modes(5)
    bins = optimal_bins(V, WAVELENGTHS, equal_widths=True)
    B = onp.asarray(bins.matrix)
    weighted = onp.sum((B - B @ V @ V.T) ** 2) / onp.sum(B**2)
    assert bins.objective == pytest.approx(weighted, rel=0.02)
    free = optimal_bins(V, WAVELENGTHS)
    assert free.objective <= bins.objective + 1e-12  # free widths never worse


def test_binning_a_fitted_spectrum_propagates_covariance():
    rng = onp.random.default_rng(1)
    V = onp.linalg.qr(rng.normal(size=(60, 4)))[0]  # rough modes: bins must still be valid
    bins = optimal_bins(V, WAVELENGTHS, n_starts=2, grid=16)
    assert onp.all(bins.widths >= onp.diff(native_edges(WAVELENGTHS))[0] * (1 - 1e-9))
    c = rng.normal(size=4)
    C = onp.diag(rng.uniform(0.5, 2.0, 4))
    reference = onp.full(60, 10.0)
    values, cov = bins.bin_modes(c, C, reference)
    onp.testing.assert_allclose(values, bins.matrix @ (reference + V @ c))
    AV = bins.matrix @ V
    onp.testing.assert_allclose(cov, AV @ C @ AV.T)
    dens, dcov = bins.bin(reference + V @ c, V @ C @ V.T, density=True)
    onp.testing.assert_allclose(dens, values / bins.widths)
    onp.testing.assert_allclose(dcov, cov / onp.outer(bins.widths, bins.widths))
    with pytest.raises(ValueError, match="orthonormal"):
        optimal_bins(2 * V, WAVELENGTHS)
