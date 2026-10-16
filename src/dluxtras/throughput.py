"""System throughput as user-supplied arrays (or an explicit two-column table).

No files are opened during import and no instrument file-naming convention is
assumed: pass wavelengths in metres and dimensionless throughput (source spectrum
excluded: optics x filter x detector QE, as appropriate for your bench).
"""

from pathlib import Path

import numpy as np

_UNITS = {"angstrom": 1e-10, "nm": 1e-9, "um": 1e-6, "m": 1.0}


def load_throughput(file, wavelength_unit="um", skiprows=0, negative_tol=1e-3):
    """Read two whitespace-delimited columns; return (n, 2) [wavelength m, throughput].

    Lines beginning with `#` are comments. No throughput renormalization.
    Slightly negative throughputs (down to `-negative_tol * peak`, i.e. table
    noise) are clipped to zero; anything more negative is an error.

    Parameters
    ----------
    file : str or path-like
        Path to the text table (wavelength, throughput).
    wavelength_unit : str, optional
        Unit of the wavelength column: one of "angstrom", "nm", "um", "m". Converted
        to metres. Default "um".
    skiprows : int, optional
        Number of leading lines to skip (passed to `numpy.loadtxt`). Default 0.
    negative_tol : float, optional
        Negative throughputs down to `-negative_tol * peak` are clipped to zero.
        Default 1e-3.

    Returns
    -------
    numpy.ndarray
        Array of shape (n, 2): wavelength in metres, dimensionless throughput.

    Raises
    ------
    ValueError
        If `wavelength_unit` is unknown, the table does not have at least two rows and
        exactly two columns, or the data are not finite, positive, increasing
        wavelengths with nonnegative (after clipping), not-all-zero throughput.
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
    """Midpoint wavelengths and integrated bin weights, with interpolated bin edges.

    The range `[wl_array[0], wl_array[-1]]` is split into `nwavels` equal-width bins.
    Throughput is linearly interpolated at the bin edges and integrated over each bin
    with the trapezoid rule, using the input samples that fall inside it.

    Parameters
    ----------
    wl_array : Array
        Finite, positive, strictly increasing wavelengths (1D, at least two samples).
    throughput_array : Array
        Nonnegative throughput, same shape as `wl_array`, not zero everywhere.
    nwavels : int
        Number of bins; a positive integer.
    norm : bool, optional
        Normalise the bin weights to sum to one. Default False.

    Returns
    -------
    wavelengths : numpy.ndarray
        Bin-centre (midpoint) wavelengths, shape (nwavels,), in the units of `wl_array`.
    weights : numpy.ndarray
        Integrated throughput per bin (throughput x wavelength units), or normalised
        weights if `norm` is True.

    Raises
    ------
    ValueError
        If the inputs fail validation, `nwavels` is not a positive integer, or the
        integrated throughput is zero.
    """
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
    """Keep where throughput > threshold * peak, plus one neighbouring sample each side.

    The kept range runs contiguously from one sample before the first sample above
    `threshold * peak` to one sample after the last (clipped to the array ends).

    Parameters
    ----------
    wavelengths : Array
        Finite, positive, strictly increasing wavelengths (1D, at least two samples).
    throughput : Array
        Nonnegative throughput, same shape as `wavelengths`, not zero everywhere.
    threshold : float or None
        Fraction of the peak throughput, with `0 <= threshold < 1`. None returns the
        validated inputs untrimmed.

    Returns
    -------
    wavelengths : numpy.ndarray
        Trimmed wavelengths.
    throughput : numpy.ndarray
        Trimmed throughput.

    Raises
    ------
    ValueError
        If the inputs fail validation or `threshold` is outside [0, 1).
    """
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

    The (optionally trimmed) range is split into `nwavels` equal-width bins and the
    integrated throughput of each bin (`filter_integrate`) is divided by the bin width.

    Parameters
    ----------
    wavelengths : Array
        Wavelengths in metres; finite, positive, strictly increasing.
    throughput : Array
        Dimensionless nonnegative throughput, same shape as `wavelengths`.
    nwavels : int, optional
        Number of wavelength bins. Default 9.
    threshold : float or None, optional
        Fraction of peak; optionally trims the band edges first, as in
        `spectrum_args` (see `trim_throughput`). Keyword-only. Default None (no trim).

    Returns
    -------
    wavelengths : numpy.ndarray
        Bin-centre wavelengths in metres, shape (nwavels,).
    throughput : numpy.ndarray
        Dimensionless mean throughput per bin, shape (nwavels,).

    Raises
    ------
    ValueError
        If the inputs, `threshold` or `nwavels` are invalid.
    """
    x, y = trim_throughput(wavelengths, throughput, threshold)
    wavels, areas = filter_integrate(x, y, nwavels, norm=False)
    bin_width = (x[-1] - x[0]) / nwavels
    return wavels, areas / bin_width


def spectrum_args(
    wavelengths, throughput, nwavels=9, *,
    photons=None, basis=None, basis_weights=None,
    threshold=1e-2,  # default 1% truncation of peak
    n_coarse=None, low_memory=False,
):
    """Keyword arguments for `CombinedBasisSpectrum(**args)`.

    The throughput is binned with `bin_throughput` (mean throughput per bin becomes
    `filt_weights`). Detected photons are `CombinedBasisSpectrum(**args).proper_flux()`.
    The same dictionary is accepted by `spectral_eigenmodes.attach_spectrum` and
    `spectral_eigenmodes.spectral_eigenmodes`.

    Parameters
    ----------
    wavelengths : Array
        Wavelengths in metres; with `throughput`, describes the system response.
    throughput : Array
        Dimensionless system throughput, same shape as `wavelengths`.
    nwavels : int, optional
        Number of wavelength bins. Default 9.
    photons : float, optional
        INCIDENT photon count (before throughput), split evenly over the bins of the
        default identity basis. Only valid when both `basis` and `basis_weights` are
        None; otherwise `basis_weights` sets the photons and passing `photons` is an
        error. Keyword-only. Default None, meaning 1.0.
    basis : Array or None, optional
        Custom basis of shape (nwavels, n_coefficients). None uses the identity
        (an independently adjustable photon count in every bin). Keyword-only.
        Default None.
    basis_weights : Array or None, optional
        Basis coefficients. Required with a custom `basis`; with the default basis,
        None gives `photons / nwavels` in every bin. Keyword-only. Default None.
    threshold : float or None, optional
        Fraction of peak throughput below which the band edges are trimmed (see
        `trim_throughput`). Keyword-only. Default 1e-2 (1% truncation of peak).
    n_coarse : int or None, optional
        Passed through (only when not None) to enable Hermite wavelength sparsity;
        see `CombinedSpectrum`. Keyword-only. Default None.
    low_memory : bool, optional
        Passed through (only when True) to enable sequential propagation; see
        `CombinedSpectrum`. Keyword-only. Default False.

    Returns
    -------
    dict
        Keys `wavels`, `filt_weights`, `basis_weights`, `basis`, plus `n_coarse` and
        `low_memory` when set.

    Raises
    ------
    ValueError
        If a custom `basis` is supplied without `basis_weights`, if `photons` is given
        together with `basis` or `basis_weights`, or if binning fails.
    """
    if photons is not None and (basis is not None or basis_weights is not None):
        raise ValueError("photons only applies to the default basis without basis_weights; "
                         "set the photons through basis_weights instead")
    photons = 1.0 if photons is None else photons
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

    The band is where throughput exceeds threshold * peak (as in `spectrum_args`); it
    is found with `trim_throughput`, so it also includes one neighbouring sample on
    each side. `lambda_c` is the midpoint of the band edges.

    Parameters
    ----------
    wavelengths : Array
        Wavelengths in metres; finite, positive, strictly increasing.
    throughput : Array
        Dimensionless nonnegative throughput, same shape as `wavelengths`.
    threshold : float or None, optional
        Fraction of peak throughput defining the band; None uses the full range.
        Keyword-only. Default 1e-2.

    Returns
    -------
    bandwidth : float
        Fractional bandwidth dlambda / lambda_c.
    centre : float
        Central wavelength lambda_c in metres.

    Raises
    ------
    ValueError
        If the inputs or `threshold` are invalid.
    """
    x, _ = trim_throughput(wavelengths, throughput, threshold)
    lo, hi = x[0], x[-1]
    centre = (lo + hi) / 2
    return float((hi - lo) / centre), float(centre)
