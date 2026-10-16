"""Throughput-weighted spectra and Hermite wavelength-sparse propagation.

`CombinedSpectrum` / `CombinedBasisSpectrum` are adapted from h-greer/hubble (spectra.py,
MIT licence, (c) 2024 h-greer); `n_coarse` (Hermite sparsity), `low_memory`, `proj` and the
incident/detected flux conventions were added here.

Notes
-----
Conventions:

- wavelengths are in metres;
- `spec_weights()` = INCIDENT photons per bin;
- `filt_weights` = mean throughput per bin;
- `proper_flux()` = DETECTED photons.
"""

import dLux as dl
import equinox as eqx
import jax
import jax.numpy as np
import numpy as numpy
from dLux.spectra import SimpleSpectrum
from jax import Array


def coarse_indices(n_wavels, n_coarse):
    """Indices of `n_coarse` evenly spread nodes in a fine grid, both ends included.

    Indices are `round(linspace(0, n_wavels - 1, n_coarse))`, de-duplicated and sorted.

    Parameters
    ----------
    n_wavels : int
        Number of wavelengths in the fine grid.
    n_coarse : int
        Number of coarse nodes; an integer value between 2 and `n_wavels` inclusive.

    Returns
    -------
    numpy.ndarray
        Integer indices into the fine grid, including 0 and `n_wavels - 1`.

    Raises
    ------
    ValueError
        If `n_coarse` is not an integer between 2 and `n_wavels`.
    """
    if int(n_coarse) != n_coarse or not 2 <= n_coarse <= n_wavels:
        raise ValueError("n_coarse must be an integer between 2 and the number of wavelengths")
    return numpy.unique(numpy.round(numpy.linspace(0, n_wavels - 1, int(n_coarse))).astype(int))


def hermite_node_coeffs(wavelengths_fine, weights, wavelengths_coarse):
    """Coefficients of coarse-node images and d/dlambda images under cubic Hermite.

    The weighted sum over the fine grid is approximated by values and wavelength
    derivatives at the coarse nodes:

    ```
    sum_j weights[j] O(lambda_fine[j]) ~= sum_k coeff_val[k] O(lambda_k)
                                          + coeff_der[k] dO/dlambda(lambda_k).
    ```

    Parameters
    ----------
    wavelengths_fine : Array
        Fine wavelengths. They must lie within the coarse range.
    weights : Array
        Weight of each fine wavelength, same shape as `wavelengths_fine`.
    wavelengths_coarse : Array
        Increasing coarse node wavelengths (at least two), same units as
        `wavelengths_fine`.

    Returns
    -------
    coeff_val : Array
        Coefficient of each coarse-node image, shape (n_coarse,).
    coeff_der : Array
        Coefficient of each coarse-node d/dlambda image, shape (n_coarse,).
    """
    wavelengths_coarse = np.asarray(wavelengths_coarse, dtype=float)
    n_coarse = wavelengths_coarse.size

    idx = np.searchsorted(wavelengths_coarse, wavelengths_fine, side="right") - 1
    idx = np.clip(idx, 0, n_coarse - 2)

    lam0 = wavelengths_coarse[idx]
    dlam = wavelengths_coarse[idx + 1] - lam0
    t = (wavelengths_fine - lam0) / dlam

    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    coeff_val = np.zeros(n_coarse, dtype=weights.dtype)
    coeff_val = coeff_val.at[idx].add(weights * h00).at[idx + 1].add(weights * h01)
    coeff_der = np.zeros(n_coarse, dtype=weights.dtype)
    coeff_der = coeff_der.at[idx].add(weights * dlam * h10).at[idx + 1].add(weights * dlam * h11)
    return coeff_val, coeff_der


def propagate_spectrum(optics, spectrum, offset, flux=1.0, return_wf=False, return_psf=False):
    """`optics.propagate` for a spectrum, honouring `spectrum.n_coarse` and `spectrum.low_memory`.

    With `n_coarse`, only the coarse wavelengths are propagated (value and
    d/dlambda in one forward-mode pass); the full grid is recovered with cubic
    Hermite interpolation of the monochromatic PSFs. With `low_memory`, nodes are
    propagated one at a time into a running sum and recomputed in the backward
    pass, so peak memory is that of ~one node (gradients ~1.5-2x slower).
    With neither this is exactly
    `optics.propagate(spectrum.wavelengths, offset, spectrum.weights * flux, ...)`.

    Parameters
    ----------
    optics : dl.BaseOpticalSystem
        Optical system to propagate through.
    spectrum : dl.spectra.BaseSpectrum
        Spectrum providing `wavelengths` and `weights`. The optional attributes
        `n_coarse` (default None) and `low_memory` (default False) are read if present,
        and `coarse_wavelengths` is used when `n_coarse` is set.
    offset : Array
        (x, y) source offset passed to the optics, in radians.
    flux : float, optional
        Multiplier applied to `spectrum.weights`. Default 1.0.
    return_wf : bool, optional
        Return the wavefront. Only allowed when neither `n_coarse` nor `low_memory`
        is set. Default False.
    return_psf : bool, optional
        Return a `dl.PSF` object (with the mean pixel scale over propagated nodes in
        the sparse/low-memory paths) instead of an image array. Default False.

    Returns
    -------
    Array or dl.PSF or Wavefront
        The polychromatic image, `dl.PSF`, or (plain path only) wavefront.

    Raises
    ------
    ValueError
        If `return_wf` is True with `n_coarse` or `low_memory` set
        ("Hermite-interpolated and low-memory spectra return PSFs, not wavefronts").
    """
    weights = spectrum.weights * flux
    n_coarse = getattr(spectrum, "n_coarse", None)
    low_memory = getattr(spectrum, "low_memory", False)
    if n_coarse is None and not low_memory:
        return optics.propagate(spectrum.wavelengths, offset, weights, return_wf, return_psf)
    if return_wf:
        raise ValueError("Hermite-interpolated and low-memory spectra return PSFs, not wavefronts")

    def mono(wavelength):
        wf = optics.propagate_mono(wavelength, offset, return_wf=True)
        return wf.psf, wf.pixel_scale

    # Each node returns images (PSF, and d/dlambda PSF for Hermite) with one coefficient each.
    if n_coarse is None:
        nodes, coeffs = spectrum.wavelengths, (weights,)

        def node(wavelength):
            psf, pixel_scale = mono(wavelength)
            return (psf,), pixel_scale
    else:
        nodes = spectrum.coarse_wavelengths
        coeffs = hermite_node_coeffs(spectrum.wavelengths, weights, nodes)

        def node(wavelength):
            (psf, pixel_scale), (dpsf, _) = jax.jvp(
                mono, (wavelength,), (np.ones_like(wavelength),)
            )
            return (psf, dpsf), pixel_scale

    if low_memory:
        def step(image, inputs):
            wavelength, coeff = inputs
            images, pixel_scale = node(wavelength)
            return image + sum(c * im for c, im in zip(coeff, images)), pixel_scale

        shape = jax.eval_shape(lambda wavelength: node(wavelength)[0][0], nodes[0])
        data, pixel_scales = jax.lax.scan(
            jax.checkpoint(step), np.zeros(shape.shape, shape.dtype), (nodes, coeffs)
        )
    else:
        images, pixel_scales = jax.vmap(node)(nodes)
        data = sum(np.tensordot(c, im, axes=1) for c, im in zip(coeffs, images))
    return dl.PSF(data, pixel_scales.mean()) if return_psf else data


class CombinedSpectrum(SimpleSpectrum):
    """Throughput-weighted spectrum with optional Hermite wavelength sparsity.

    Base class: subclasses define `spec_weights()` (INCIDENT photons per bin); this
    class combines it with `filt_weights` (mean throughput per bin).
    `n_coarse=None` propagates every wavelength; an integer propagates only
    `n_coarse` evenly spread nodes of the grid and Hermite-interpolates the rest.
    `low_memory=True` propagates sequentially (either way) to cut peak memory.
    Both are used by `HermitePointSource` / `PointSourceArcsec` via `propagate_spectrum`;
    see `recommend_wavelength_sampling` for choosing `n_coarse` (n_coarse ~ 2K).

    Parameters
    ----------
    wavels : Array
        Wavelengths in metres (1D).
    filt_weights : Array
        Mean throughput per bin (dimensionless), same shape as `wavels`.
    basis_weights : Array
        Spectral coefficients, stored as floats; interpreted by subclasses.
    n_coarse : int or None, optional
        Number of coarse Hermite nodes, an integer between 2 and the number of
        wavelengths, or None to propagate every wavelength. Default None.
    low_memory : bool, optional
        Propagate nodes sequentially to cut peak memory. Default False.

    Raises
    ------
    ValueError
        If `filt_weights` and `wavels` shapes differ, or `n_coarse` is invalid.
    """

    filt_weights: Array
    basis_weights: Array
    n_coarse: int | None = eqx.field(static=True)
    low_memory: bool = eqx.field(static=True)

    def __init__(self, wavels, filt_weights, basis_weights, n_coarse=None, low_memory=False):
        super().__init__(wavels)
        self.filt_weights = np.asarray(filt_weights, float)
        self.basis_weights = np.asarray(basis_weights, dtype=float)
        if self.filt_weights.shape != self.wavelengths.shape:
            raise ValueError("Filter weights and wavelengths must have matching shapes")
        if n_coarse is not None:
            coarse_indices(self.wavelengths.size, n_coarse)
            n_coarse = int(n_coarse)
        self.n_coarse = n_coarse
        self.low_memory = bool(low_memory)

    @property
    def coarse_wavelengths(self):
        """Wavelengths actually propagated (all of them when `n_coarse` is None).

        Returns
        -------
        Array
            Coarse node wavelengths in metres, selected by `coarse_indices`.
        """
        if self.n_coarse is None:
            return self.wavelengths
        return self.wavelengths[coarse_indices(self.wavelengths.size, self.n_coarse)]

    def spec_weights(self):
        """INCIDENT photons per wavelength bin (before throughput).

        Must be implemented by subclasses.

        Returns
        -------
        Array
            Incident photons per bin, same shape as `wavelengths`.

        Raises
        ------
        NotImplementedError
            Always, on the base class ("Use a concrete CombinedSpectrum subclass").
        """
        raise NotImplementedError("Use a concrete CombinedSpectrum subclass")

    @property
    def weights(self):
        """Normalised DETECTED photon fraction per bin.

        Returns
        -------
        Array
            `filt_weights * spec_weights()`, divided by its sum (sums to one).
        """
        weights = self.filt_weights * self.spec_weights()
        return weights / weights.sum()

    @property
    def flux(self):
        """Total INCIDENT photons, `sum(spec_weights())` (before throughput).

        Returns
        -------
        Array
            Scalar total incident photon count.
        """
        return np.sum(self.spec_weights())

    def proper_flux(self):
        """Total DETECTED photons, `sum(spec_weights() * filt_weights)`.

        Returns
        -------
        Array
            Scalar total detected photon count.
        """
        return np.sum(self.spec_weights() * self.filt_weights)

    def normalise(self):
        """Return the spectrum unchanged.

        `weights` is already normalised, so no rescaling is needed.

        Returns
        -------
        CombinedSpectrum
            `self`.
        """
        return self


class CombinedBasisSpectrum(CombinedSpectrum):
    """`CombinedSpectrum` whose incident photons per bin are a linear basis expansion.

    `spec_weights()` is `maximum(basis @ (proj @ basis_weights), 1e-08)`, i.e. INCIDENT
    photons per bin, clipped below at 1e-08. `throughput.spectrum_args` builds the
    keyword arguments for this class.

    Parameters
    ----------
    wavels : Array
        Wavelengths in metres (1D).
    filt_weights : Array
        Mean throughput per bin (dimensionless), same shape as `wavels`.
    basis_weights : Array
        Basis coefficients, shape (n_coefficients,).
    basis : Array
        Basis vectors as columns, shape (n_wavelengths, n_coefficients). Stored as
        `basis_vects`.
    proj : Array or None, optional
        Square (n_coefficients, n_coefficients) matrix applied to `basis_weights`
        before the basis. None uses the identity. Default None.
    n_coarse : int or None, optional
        Number of coarse Hermite nodes (see `CombinedSpectrum`). Default None.
    low_memory : bool, optional
        Propagate nodes sequentially to cut peak memory. Default False.

    Raises
    ------
    ValueError
        If `basis` does not have shape (n_wavelengths, n_coefficients), `proj` is not
        square with one row per basis coefficient, or the `CombinedSpectrum` checks fail.
    """

    basis_vects: Array
    proj: Array

    def __init__(
        self, wavels, filt_weights, basis_weights, basis, proj=None, n_coarse=None,
        low_memory=False,
    ):
        self.basis_vects = np.asarray(basis)
        super().__init__(wavels, filt_weights, basis_weights, n_coarse, low_memory)
        if proj is None:
            self.proj = np.eye(self.basis_weights.shape[0])
        else:
            self.proj = np.asarray(proj)
        if self.basis_vects.shape != (len(self.wavelengths), len(self.basis_weights)):
            raise ValueError("basis must have shape (n_wavelengths, n_coefficients)")
        if self.proj.shape != (len(self.basis_weights), len(self.basis_weights)):
            raise ValueError("proj must be square with one row per basis coefficient")

    def spec_weights(self):
        """INCIDENT photons per wavelength bin (before throughput).

        Returns
        -------
        Array
            `maximum(basis_vects @ (proj @ basis_weights), 1e-08)`, one value per bin.
        """
        return np.maximum(self.basis_vects @ (self.proj @ self.basis_weights), 1e-08)


def recommend_wavelength_sampling(K, photons, *, jvp_cost=2.5, min_speedup=1.5, verbose=True):
    """Print and return recommended `n_coarse` / `n_fine` from the number of constrainable modes.

    One Hermite node costs ~`jvp_cost` exact wavelengths, so the estimated speedup is
    `n_exact / (jvp_cost * n_coarse)`; below `min_speedup` the recommendation is plain
    propagation (`n_coarse=None`) with `n_exact` wavelengths. Fine wavelengths are ~free
    with Hermite, so `n_fine = max(300, 20K)` there.

    Parameters
    ----------
    K : int
        Number of constrainable spectral modes, 1 / (|a_0| sqrt(d_i)) < 0.1, from a
        Fisher matrix with >= ~100 fine wavelengths (K is unconverged for wide filters
        at ~30); see `spectral_eigenmodes.constrainable_modes`. Must be >= 1.
    photons : float
        Total detected photons N. Must be positive.
    jvp_cost : float, optional
        Cost of one Hermite node in units of exact wavelengths. Keyword-only.
        Default 2.5.
    min_speedup : float, optional
        Minimum estimated speedup for recommending sparse propagation. Keyword-only.
        Default 1.5.
    verbose : bool, optional
        Print the recommendation. Keyword-only. Default True.

    Returns
    -------
    dict
        Keys `speedup` (float, estimated speedup), `sparse` (bool, whether sparse
        propagation is recommended), `n_coarse` (`max(3, 2K)` if sparse, else None) and
        `n_fine` (`max(300, 20K)` if sparse, else `n_exact`).

    Raises
    ------
    ValueError
        If `K` is not a positive integer, or `photons <= 0`.

    Notes
    -----
    Empirical rules, fitted on JWST/NIRCam models (point source, Poisson noise, model error
    Delta chi^2 < 1; 4 filters, 32-256 px stamps, N = 1e8-1e10). On those models, and
    re-checked on generic optics and synthetic filters by `validation/wavelength_sampling.py`
    (96 cases; see the "Wavelength sampling" page of the documentation), neither rule ever
    undershot:

    ```
    Hermite: n_coarse = max(3, 2K)
    exact:   n_exact  = ceil(23.5 K^0.553 (N/1e9)^0.208)
    ```
    """
    if int(K) != K or K < 1 or photons <= 0:
        raise ValueError("K must be a positive integer (a number of modes) and photons positive")
    K = int(K)
    n_coarse = max(3, 2 * K)
    n_exact = int(numpy.ceil(23.5 * K**0.553 * (photons / 1e9) ** 0.208))
    speedup = n_exact / (jvp_cost * n_coarse)
    sparse = speedup >= min_speedup
    result = dict(
        speedup=float(speedup), sparse=sparse,
        n_coarse=n_coarse if sparse else None,
        n_fine=max(300, 20 * K) if sparse else n_exact,
    )
    if verbose:
        print(f"K = {K}, N = {photons:.1e}: Hermite n_coarse = {n_coarse}; "
              f"exact n_fine = {n_exact}; estimated speedup = {speedup:.1f}x")
        if sparse:
            print(f"  -> use sparse propagation: n_coarse={n_coarse}, n_fine={result['n_fine']}")
        else:
            print(f"  -> speedup below {min_speedup}x: use n_coarse=None, n_fine={n_exact}"
                  " (low_memory=True if memory is tight)")
    return result
