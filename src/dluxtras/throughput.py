"""System throughput as user-supplied arrays (or an explicit two-column table).

No files are opened during import and no instrument file-naming convention is
assumed: pass wavelengths [m] and dimensionless throughput (source spectrum
excluded: optics x filter x detector QE, as appropriate for your bench).
"""

from pathlib import Path

import numpy as np

_UNITS = {"angstrom": 1e-10, "nm": 1e-9, "um": 1e-6, "m": 1.0}


def load_throughput(file, wavelength_unit="um", skiprows=0, negative_tol=1e-3):
    """Read two whitespace-delimited columns; return (n, 2) [wavelength m, throughput].

    Lines beginning with # are comments. No throughput renormalization.
    Slightly negative throughputs (down to -negative_tol * peak, i.e. table
    noise) are clipped to zero; anything more negative is an error.
    """
    if wavelength_unit not in _UNITS:
        raise ValueError(f"wavelength_unit must be one of {tuple(_UNITS)}")
    table = np.loadtxt(Path(file), comments="#", ndmin=2, skiprows=skiprows)
    if table.shape[1] != 2 or table.shape[0] < 2:
        raise ValueError("Expected at least two rows and exactly two columns")
    table = table.copy()
    table[:, 0] *= _UNITS[wavelength_unit]
    tiny = (table[:, 1] < 0) & (table[:, 1] >= -negative_tol * table[:, 1].max())
    table[tiny, 1] = 0.0
    _check(table[:, 0], table[:, 1])
    return table


def _check(wavelengths, throughput):
    x, y = np.asarray(wavelengths, float), np.asarray(throughput, float)
    if (
        x.ndim != 1
        or x.size < 2
        or y.shape != x.shape
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any(x <= 0)
        or np.any(np.diff(x) <= 0)
        or np.any(y < 0)
    ):
        raise ValueError(
            "Need finite, positive, increasing wavelengths and matching nonnegative throughput"
        )
    if not np.any(y > 0):
        raise ValueError("Throughput is zero everywhere")
    return x, y


def filter_integrate(wl_array, throughput_array, nwavels, norm=False):
    """Midpoint wavelengths and integrated bin weights, with interpolated bin edges."""
    x, y = _check(wl_array, throughput_array)
    if int(nwavels) != nwavels or nwavels < 1:
        raise ValueError("nwavels must be a positive integer")
    edges = np.linspace(x[0], x[-1], int(nwavels) + 1)
    areas = []
    for left, right in zip(edges[:-1], edges[1:]):
        samples = np.concatenate(([left], x[(x > left) & (x < right)], [right]))
        values = np.interp(samples, x, y)
        # np.trapezoid is absent in NumPy 1.x.
        areas.append(np.sum(np.diff(samples) * (values[:-1] + values[1:]) / 2))
    weights = np.array(areas)
    if weights.sum() <= 0:
        raise ValueError("Integrated throughput is zero")
    if norm:
        weights /= weights.sum()
    return (edges[:-1] + edges[1:]) / 2, weights


def trim_throughput(wavelengths, throughput, threshold):
    """Keep where throughput > threshold * peak, plus one neighbouring sample each side."""
    x, y = _check(wavelengths, throughput)
    if threshold is None:
        return x, y
    if not 0 <= threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    keep = np.flatnonzero(y > threshold * y.max())
    lo, hi = max(0, keep[0] - 1), min(x.size, keep[-1] + 2)
    return x[lo:hi], y[lo:hi]


def bin_throughput(wavelengths, throughput, nwavels=9, *, threshold=None):
    """Return bin-centre wavelengths [m] and dimensionless MEAN throughput per bin.

    wavelengths are in metres. threshold (fraction of peak) optionally trims
    the band edges first, as in spectrum_args.
    """
    x, y = trim_throughput(wavelengths, throughput, threshold)
    wavels, areas = filter_integrate(x, y, nwavels, norm=False)
    bin_width = (x[-1] - x[0]) / nwavels
    return wavels, areas / bin_width


def spectrum_args(
    wavelengths, throughput, nwavels=9, *,
    photons=1.0, basis=None, basis_weights=None,
    threshold=1e-2,  # default 1% truncation of peak
    n_coarse=None, low_memory=False,
):
    """Keyword arguments for CombinedBasisSpectrum(**args).

    wavelengths [m] and throughput describe the system response. photons is
    the INCIDENT photon count (before throughput), split evenly over the bins
    of the default identity basis; detected photons are
    CombinedBasisSpectrum(**args).proper_flux(). n_coarse and low_memory are
    passed through (only when set) to enable Hermite wavelength sparsity /
    sequential propagation; see CombinedSpectrum.
    """
    wavels, filt = bin_throughput(wavelengths, throughput, nwavels, threshold=threshold)

    # Default: independently adjustable photon count in every bin.
    if basis is None:
        basis = np.eye(nwavels)
        if basis_weights is None:
            basis_weights = np.full(nwavels, photons / nwavels)
    elif basis_weights is None:
        raise ValueError("Supply basis_weights when supplying a custom basis")

    args = dict(wavels=wavels, filt_weights=filt, basis_weights=basis_weights, basis=basis)
    if n_coarse is not None:
        args["n_coarse"] = n_coarse
    if low_memory:
        args["low_memory"] = True
    return args


def fractional_bandwidth(wavelengths, throughput, *, threshold=1e-2):
    """Return (fractional bandwidth dlambda / lambda_c, central wavelength [m]).

    The band is where throughput exceeds threshold * peak (as in spectrum_args).
    """
    x, _ = trim_throughput(wavelengths, throughput, threshold)
    lo, hi = x[0], x[-1]
    centre = (lo + hi) / 2
    return float((hi - lo) / centre), float(centre)
