"""Small spectral Fisher eigenbases for existing dLux optics/telescopes.

Enable `jax_enable_x64` before constructing models. Dependencies: dLux, JAX,
NumPy. No files are loaded here. Pass the dictionary from `throughput.spectrum_args`.

Coordinates are incident photons per wavelength bin, before `filt_weights`.
Eigenvectors are Euclidean unit-norm columns in that coordinate system.
This metric depends on wavelength sampling; no bin-width whitening is applied.
Other model parameters (e.g. source position, pixel scale) are held fixed unless
listed in `marginalise` / `marginalise_shared`, in which case they are marginalised
(fitted freely); see `dluxtras.eigenmodes`. The API matches
`dluxtras.wavefront_eigenmodes`. Poisson noise requires predictions in expected counts,
not MJy/sr.

Notes
-----
No Hessian is needed: cached first derivatives form the expected Fisher matrix. Source
classes inherit dLux's Zodiax model behaviour. Detector covariance between pixels is
not modelled.

Examples
--------
In your notebook:

```python
import jax
jax.config.update('jax_enable_x64', True)
from dluxtras.throughput import spectrum_args
from dluxtras.spectral_eigenmodes import spectral_eigenmodes, constrainable_modes
args = spectrum_args(wavelengths_m, throughput, nwavels=30, photons=1e6)
modes = spectral_eigenmodes(telescope, args, k=30, noise='poisson')
K = constrainable_modes(modes)   # 1/(|a0| sqrt(d_i)) < 0.1
# Bare optics also work; optionally pass detector=your_detector.
# Gaussian alternative: noise='gaussian', variance=1.0
# variance_fn=lambda mu: mu + read_noise**2 supports mixed noise.
# bad_masks=image.bad excludes the same pixels as your image loader.
# modes.vectors: (30, 30); modes.values: (30,); modes.sigma: (30,)

# Marginalise the source position and the pixel scale:
modes = spectral_eigenmodes(telescope, args, k=30,
                            marginalise=['source.position'],
                            marginalise_shared=['optics.psf_pixel_scale'])
```
"""

import warnings
from dataclasses import dataclass

import dLux as dl
import dLux.utils as dlu
import jax
import jax.numpy as np
import numpy as onp

from .eigenmodes import _each, _Linearisation, _schur_terms, leading_modes, noise_weights

__all__ = ["SpectralSource", "attach_spectrum", "spectral_fisher",
           "leading_modes", "spectral_eigenmodes",
           "SpectralModes", "mode_criterion", "constrainable_modes"]


class SpectralSource(dl.PointSource):
    """Point source with absolute bin fluxes; inherited position is in radians.

    The image is `sum_i bin_flux[i] * filt_weights[i] * PSF(wavelengths[i])`, with each
    monochromatic PSF propagated with unit weight. `bin_flux` is in incident photons per
    wavelength bin, before `filt_weights`.

    Signed `bin_flux` is allowed for local linear perturbations. A physical
    reference must be nonnegative. No positivity clipping is differentiated.
    `flux` is inherited for compatibility but is not used as another scale.

    Parameters
    ----------
    wavels : Array
        Wavelengths in metres (1D).
    filt_weights : Array
        Throughput per bin, same shape as `wavels`.
    bin_flux : Array
        Incident photons per wavelength bin, same shape as `wavels`.
    position_arcsec : Array or tuple, optional
        (x, y) position in arcseconds; converted to radians and stored as `position`.
        Default (0., 0.).

    Raises
    ------
    ValueError
        If `wavels`, `filt_weights` and `bin_flux` do not have matching 1D shapes.
    """
    bin_flux: jax.Array
    filt_weights: jax.Array

    def __init__(self, wavels, filt_weights, bin_flux, position_arcsec=(0., 0.)):
        super().__init__(wavelengths=np.asarray(wavels, dtype=float),
                         position=dlu.arcsec2rad(np.asarray(position_arcsec, dtype=float)))
        self.bin_flux = np.asarray(bin_flux, dtype=float)
        self.filt_weights = np.asarray(filt_weights, dtype=float)
        if self.bin_flux.shape != self.wavelengths.shape or self.filt_weights.shape != self.wavelengths.shape:
            raise ValueError("wavels, filt_weights and bin_flux must have matching 1D shapes")

    def monochromatic_psf(self, optics, wavelength):
        """Unit-weight PSF of the source at a single wavelength.

        Parameters
        ----------
        optics : dl.BaseOpticalSystem
            Optical system to propagate through.
        wavelength : Array
            Scalar wavelength in metres.

        Returns
        -------
        dl.PSF
            PSF from `optics.propagate` at `self.position` with weight one.
        """
        return optics.propagate(wavelength[None], self.position,
                                weights=np.ones(1), return_psf=True)

    def model(self, optics, return_wf=False, return_psf=False):
        """Incoherent sum of monochromatic PSFs weighted by `bin_flux * filt_weights`.

        Wavelengths are propagated one at a time (`jax.lax.scan`).

        Parameters
        ----------
        optics : dl.BaseOpticalSystem
            Optical system to propagate through.
        return_wf : bool, optional
            Not supported; must be False. Default False.
        return_psf : bool, optional
            Return a `dl.PSF` (with the pixel scale of the first wavelength's PSF)
            instead of an image array. Default False.

        Returns
        -------
        Array or dl.PSF
            The polychromatic image, or a `dl.PSF` if `return_psf` is True.

        Raises
        ------
        ValueError
            If `return_wf` is True.
        """
        if return_wf:
            raise ValueError("This incoherent spectral source returns images/PSFs, not wavefronts")
        template = self.monochromatic_psf(optics, self.wavelengths[0])

        def add(image, inputs):
            wavelength, weight = inputs
            psf = self.monochromatic_psf(optics, wavelength)
            return image + weight * psf.data, None

        data, _ = jax.lax.scan(add, np.zeros_like(template.data),
                              (self.wavelengths, self.bin_flux * self.filt_weights))
        return template.set("data", data) if return_psf else data


def _reference(args):
    wavels = onp.asarray(args["wavels"], dtype=float)
    filt = onp.asarray(args["filt_weights"], dtype=float)
    if wavels.ndim != 1 or not wavels.size or filt.shape != wavels.shape:
        raise ValueError("Need matching 1D wavels and filt_weights")
    if not onp.isfinite(wavels).all() or onp.any(wavels <= 0) or onp.any(onp.diff(wavels) <= 0):
        raise ValueError("Wavelengths must be finite, positive and increasing")
    if not onp.isfinite(filt).all() or onp.any(filt < 0) or not onp.any(filt > 0):
        raise ValueError("Throughput must be finite, nonnegative and nonzero")
    basis = onp.asarray(args["basis"], dtype=float)
    coeff = onp.asarray(args["basis_weights"], dtype=float)
    if coeff.ndim != 1 or basis.shape != (wavels.size, coeff.size):
        raise ValueError("basis must have shape (n_wavelengths, n_coefficients)")
    proj = onp.asarray(args.get("proj", onp.eye(coeff.size)), dtype=float)
    if proj.shape != (coeff.size, coeff.size):
        raise ValueError("proj must have shape (n_coefficients, n_coefficients)")
    # Same reference convention as the supplied spectra.py; clipping is used
    # only to evaluate the reference, never to define the Fisher coordinates.
    q0 = onp.maximum(basis @ (proj @ coeff), 1e-8)
    if not onp.isfinite(q0).all():
        raise ValueError("Nonfinite reference spectrum")
    return wavels, filt, q0


def attach_spectrum(model_or_optics, spectrum_args, *, detector=None,
                    position_arcsec=(0., 0.)):
    """Replace the source, preserving existing telescope optics and detector.

    The new source is a `SpectralSource` whose `bin_flux` is the reference incident
    spectrum `maximum(basis @ (proj @ basis_weights), 1e-8)` from `spectrum_args`
    (clipping only evaluates the reference; it does not define Fisher coordinates).

    Parameters
    ----------
    model_or_optics : dl.Telescope or dl.BaseOpticalSystem or list or tuple
        A telescope-like model (anything with `optics` and `source` attributes) or
        bare optics, or a list/tuple of these. Bare optics create a `dl.Telescope`.
        Lists/tuples return a list of telescopes with one shared reference spectrum.
    spectrum_args : dict
        Dictionary from `throughput.spectrum_args`: `wavels` (metres), `filt_weights`,
        `basis`, `basis_weights` and optionally `proj`.
    detector : dl.BaseDetector or None, optional
        A non-None detector overrides the existing detector (or is used for new
        telescopes). Keyword-only. Default None.
    position_arcsec : Array or tuple, optional
        Source position in arcseconds; defaults to field centre. For a list it may
        also be (n_models, 2), one position per model. Keyword-only. Default (0., 0.).

    Returns
    -------
    dl.Telescope or list
        The model with the new source, or a list of models for list/tuple input.

    Raises
    ------
    ValueError
        If the spectrum arguments are invalid (wavelengths not matching 1D, finite,
        positive and increasing; throughput not finite, nonnegative and nonzero;
        `basis` not (n_wavelengths, n_coefficients); `proj` not
        (n_coefficients, n_coefficients); or a nonfinite reference spectrum).
    """
    wavels, filt, q0 = _reference(spectrum_args)

    def attach(item, position):
        source = SpectralSource(wavels, filt, q0, position)
        if hasattr(item, "optics") and hasattr(item, "source"):
            model = item.set("source", source)
            return model if detector is None else model.set("detector", detector)
        return dl.Telescope(optics=item, source=source, detector=detector)

    if isinstance(model_or_optics, (list, tuple)):
        positions = onp.broadcast_to(onp.asarray(position_arcsec, dtype=float),
                                     (len(model_or_optics), 2))
        return [attach(item, position) for item, position in zip(model_or_optics, positions)]
    return attach(model_or_optics, position_arcsec)


def spectral_fisher(models, *, noise="poisson", variance=1.0, variance_fn=None,
                    bad_masks=None, marginalise=None, marginalise_shared=None, rcond=1e-10):
    """Dense Fisher matrix of the bin fluxes, optionally with nuisances marginalised.

    The Fisher matrix is `sum J.T diag(1 / variance) J` over models and included
    pixels, where `J` is the Jacobian of the (detected) image with respect to
    `bin_flux`, the incident photons per wavelength bin. Monochromatic optical
    responses are computed one at a time. The detector is differentiated at the full
    polychromatic reference using JVP, so a differentiable nonlinear detector is handled
    locally too.

    With `marginalise` or `marginalise_shared`, it is the Fisher matrix left once those
    nuisance parameters are fitted freely (the Schur complement); see
    `dluxtras.eigenmodes`. Other parameters are held fixed.

    Parameters
    ----------
    models : dl.Telescope or list
        A model, or list/tuple of models, prepared with `attach_spectrum`. All must share
        wavelengths and reference bin fluxes.
    noise : {"poisson", "gaussian"}, optional
        `"poisson"` (default) uses the reference image as the variance, so it must be in
        counts and strictly positive on included pixels. `"gaussian"` uses `variance`.
    variance : float or Array or list, optional
        Gaussian variance (not standard deviation): a scalar or image-shaped array, used
        for every model, or a list with one entry per model. Ignored for Poisson noise or
        when `variance_fn` is given. Default 1.
    variance_fn : callable, optional
        `variance_fn(mu)` overrides either noise choice and is frozen at the reference
        image `mu`, e.g. `lambda mu: mu + read_noise**2`.
    bad_masks : Array or list, optional
        Boolean image mask (True excludes a pixel) used for every model, or a list with
        one mask (or None) per model.
    marginalise : list, optional
        Per-model nuisances: model keys (e.g. `"source.position"`) and/or arrays of
        directions in bin-flux space, shape `(n_bins,)` or `(n_bins, m)`. Each model gets
        its own copy. Default None.
    marginalise_shared : list, optional
        Nuisances shared by all models, e.g. `["optics.psf_pixel_scale"]`. Default None.
    rcond : float, optional
        Relative cutoff of the nuisance pseudo-inverse (in unit-free form). Default 1e-10.

    Returns
    -------
    Array
        Symmetric Fisher matrix, shape `(n_bins, n_bins)`.

    Raises
    ------
    ValueError
        If `jax_enable_x64` is not enabled; `noise` is invalid; a model has no
        `SpectralSource` (use `attach_spectrum` first); the number of masks or variances
        does not match the number of models; models do not share wavelengths and
        reference bin fluxes; a mask does not match the image or excludes every pixel;
        the reference image is nonfinite (or negative for Poisson) on included pixels;
        or included variances are not finite and positive (mask zero-mean Poisson pixels
        explicitly).
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before building models")
    models = list(models) if isinstance(models, (list, tuple)) else [models]
    if not models or not all(isinstance(m.source, SpectralSource) for m in models):
        raise ValueError("Use attach_spectrum first")
    masks = _each(bad_masks, len(models), "bad_masks")
    variances = _each(variance, len(models), "variance")
    reference = models[0].source
    n = reference.bin_flux.size
    F = onp.zeros((n, n))
    lins = []
    for model, bad, model_variance in zip(models, masks, variances):
        source, optics = model.source, model.optics
        if not onp.array_equal(source.wavelengths, reference.wavelengths) or not \
                onp.array_equal(source.bin_flux, reference.bin_flux):
            raise ValueError("Models must share wavelengths and reference bin fluxes")
        mono = jax.jit(lambda wavelength: source.monochromatic_psf(optics, wavelength))
        psfs = [mono(w) for w in source.wavelengths]
        responses = np.stack([p.data for p in psfs]) * source.filt_weights[:, None, None]
        optical = np.tensordot(source.bin_flux, responses, axes=1)
        template = psfs[0]

        def detect(data):
            return data if model.detector is None else model.detector.model(
                template.set("data", data))

        mu, pushforward = jax.linearize(detect, optical)
        weight = noise_weights(mu, noise=noise, variance=model_variance,
                               variance_fn=variance_fn, bad_mask=bad).ravel()
        column = jax.jit(lambda tangent: pushforward(tangent).ravel())
        J = np.stack([column(response) for response in responses], axis=1)
        F = F + onp.asarray(J.T @ (J * weight[:, None]))
        lins.append(_Linearisation(model, weight, lambda v, J=J: J @ v,
                                   lambda t, J=J: J.T @ t))
    terms = _schur_terms(lins, n, marginalise, marginalise_shared, rcond)
    if terms is not None:
        G, C_pinv = terms
        F = F - G @ C_pinv @ G.T
    return (F + F.T) / 2


@dataclass
class SpectralModes:
    """Result of `spectral_eigenmodes`: the Fisher matrix and its leading eigenmodes.

    All quantities are in incident-photons-per-bin coordinates (before `filt_weights`).
    The fields shared with `wavefront_eigenmodes.WavefrontModes` (`reference`,
    `fisher`, `values`, `vectors`, `residuals`, `sigma`, `coefficients`) mean the same.

    Attributes
    ----------
    wavelengths : Array
        Bin wavelengths in metres, shape (n_bins,).
    filt_weights : Array
        Throughput per bin, shape (n_bins,).
    reference : Array
        Reference incident photons per bin (`bin_flux` of the source), shape (n_bins,).
    fisher : Array
        Fisher matrix (marginalised, if requested), shape (n_bins, n_bins).
    values : Array
        Leading Fisher eigenvalues in descending order, shape (k,); numerically null
        eigenvalues are set to zero.
    vectors : Array
        Corresponding Euclidean unit-norm eigenvectors as columns, shape (n_bins, k).
    residuals : Array
        Relative eigen-residuals, shape (k,); ~1e-15 for this dense calculation.
    """

    wavelengths: jax.Array
    filt_weights: jax.Array
    reference: jax.Array
    fisher: jax.Array
    values: jax.Array
    vectors: jax.Array
    residuals: jax.Array = None

    @property
    def sigma(self):
        """1-sigma error of each mode's coefficient, in incident photons.

        This is conditional on the other modes being known (all other parameters held
        fixed, or marginalised if requested).

        Returns
        -------
        Array
            `1 / sqrt(values)`, with inf where the eigenvalue is not positive; shape (k,).
        """
        positive = self.values > 0
        return np.where(positive, 1 / np.sqrt(np.where(positive, self.values, 1.)), np.inf)

    def coefficients(self, bin_flux):
        """Project a spectrum onto the modes; useful as `basis_weights`.

        Parameters
        ----------
        bin_flux : Array
            Incident photons per bin, shape (n_bins,).

        Returns
        -------
        Array
            `vectors.T @ bin_flux`, shape (k,).
        """
        return self.vectors.T @ np.asarray(bin_flux)


def spectral_eigenmodes(model_or_optics, spectrum_args, k=6, *, detector=None,
                        position_arcsec=(0., 0.), marginalise=None, marginalise_shared=None,
                        noise="poisson", variance=1.0, variance_fn=None, bad_masks=None,
                        rcond=1e-10):
    """Leading spectral Fisher eigenmodes in one call.

    Runs `attach_spectrum`, `spectral_fisher` and `leading_modes` in turn. The arguments
    match `wavefront_eigenmodes.wavefront_eigenmodes` where they overlap.

    Parameters
    ----------
    model_or_optics : dl.Telescope or dl.BaseOpticalSystem or list or tuple
        Model(s) or bare optics; see `attach_spectrum`.
    spectrum_args : dict
        Dictionary from `throughput.spectrum_args`.
    k : int, optional
        Number of leading modes to return. Default 6.
    detector : dl.BaseDetector, optional
        Detector override; see `attach_spectrum`.
    position_arcsec : Array or tuple, optional
        Source position(s) in arcseconds; see `attach_spectrum`. Default (0., 0.).
    marginalise, marginalise_shared : list, optional
        Nuisances to marginalise, per model or shared; see `spectral_fisher`. For
        example `marginalise_shared=["optics.psf_pixel_scale"]`.
    noise, variance, variance_fn, bad_masks
        Noise model; see `spectral_fisher`.
    rcond : float, optional
        See `spectral_fisher`. Default 1e-10.

    Returns
    -------
    SpectralModes

    Raises
    ------
    ValueError
        As raised by `attach_spectrum`, `spectral_fisher` or `leading_modes`.
    """
    models = attach_spectrum(model_or_optics, spectrum_args, detector=detector,
                             position_arcsec=position_arcsec)
    fisher = spectral_fisher(models, noise=noise, variance=variance, variance_fn=variance_fn,
                             bad_masks=bad_masks, marginalise=marginalise,
                             marginalise_shared=marginalise_shared, rcond=rcond)
    values, vectors, residuals = leading_modes(fisher, k)
    source = (models[0] if isinstance(models, list) else models).source
    return SpectralModes(source.wavelengths, source.filt_weights, source.bin_flux,
                         fisher, values, vectors, residuals)


def mode_criterion(modes):
    """Fractional modal uncertainty 1 / (|a0| sqrt(d_i)) for every computed mode.

    d_i are the Fisher eigenvalues (`modes.values`) and a0 = v_0 . q0 is the
    reference spectrum's coefficient on the LEADING mode, both in the
    incident-photons-per-bin coordinates of `spectral_fisher`. The value scales
    as N**-1/2 with the detected photon count N = reference image sum, so
    build the Fisher at the real flux. Null modes (d_i = 0) give inf.

    Parameters
    ----------
    modes : SpectralModes
        Output of `spectral_eigenmodes`.

    Returns
    -------
    numpy.ndarray
        1 / (|a0| sqrt(d_i)) per mode, shape (k,); inf for non-positive d_i.

    Raises
    ------
    ValueError
        If the reference spectrum has no component on the leading mode (a0 = 0).
    """
    values = onp.asarray(modes.values, dtype=float)
    a0 = abs(float(onp.asarray(modes.vectors)[:, 0] @ onp.asarray(modes.reference)))
    if a0 == 0:
        raise ValueError("Reference spectrum has no component on the leading mode")
    out = onp.full(values.shape, onp.inf)
    positive = values > 0
    out[positive] = 1 / (a0 * onp.sqrt(values[positive]))
    return out


def constrainable_modes(modes, threshold=0.1):
    """K = number of modes with 1 / (|a0| sqrt(d_i)) < threshold (default 0.1).

    Compute modes with k equal to the number of wavelength bins (and >= ~100
    bins for wide bands, otherwise K is unconverged). If every computed mode
    passes, K is only a lower bound and a warning is issued. Use K in
    `dluxtras.spectra.recommend_wavelength_sampling` (n_coarse ~ 2K).

    Parameters
    ----------
    modes : SpectralModes
        Output of `spectral_eigenmodes`.
    threshold : float, optional
        Positive threshold on the criterion from `mode_criterion`. Default 0.1.

    Returns
    -------
    int
        K, the number of modes below `threshold`.

    Raises
    ------
    ValueError
        If `threshold` is not positive, or as raised by `mode_criterion`.

    Warns
    -----
    UserWarning
        If every computed mode passes, so K is only a lower bound.
    """
    if not 0 < threshold:
        raise ValueError("threshold must be positive")
    passed = mode_criterion(modes) < threshold
    count = int(passed.sum())
    if count == passed.size:
        warnings.warn("All computed modes are constrainable: K is a lower bound; "
                      "compute more modes or wavelength bins.", stacklevel=2)
    return count
