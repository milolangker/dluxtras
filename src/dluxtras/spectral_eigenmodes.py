"""Small spectral Fisher eigenbases for existing dLux optics/telescopes.

Enable jax_enable_x64 before constructing models. Dependencies: dLux, JAX,
NumPy. No files are loaded here. Pass the dictionary from throughput.spectrum_args.

Coordinates are incident photons per wavelength bin, before filt_weights.
Eigenvectors are Euclidean unit-norm columns in that coordinate system.
This metric depends on wavelength sampling; no bin-width whitening is applied.
All other parameters (including source position and wavefront) are fixed.
Poisson noise requires predictions in expected counts, not MJy/sr.

Example in your notebook:
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
    # modes.vectors: (30, 6); modes.values: (6,); modes.sigma: (6,)

No Hessian is needed: cached first derivatives form the expected Fisher,
as in the supplied wavefront/image modules. Source classes inherit dLux's
Zodiax model behavior. Detector covariance between pixels is not modeled.
"""

import warnings
from dataclasses import dataclass

import dLux as dl
import dLux.utils as dlu
import jax
import jax.numpy as np
import numpy as onp

__all__ = ["SpectralSource", "attach_spectrum", "spectral_fisher",
           "leading_modes", "spectral_eigenmodes",
           "SpectralModes", "mode_criterion", "constrainable_modes"]


class SpectralSource(dl.PointSource):
    """Point source with absolute bin fluxes; inherited position is in radians.

    Signed bin_flux is allowed for local linear perturbations. A physical
    reference must be nonnegative. No positivity clipping is differentiated.
    flux is inherited for compatibility but is not used as another scale.
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
        return optics.propagate(wavelength[None], self.position,
                                weights=np.ones(1), return_psf=True)

    def model(self, optics, return_wf=False, return_psf=False):
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

    Bare optics create a dl.Telescope. Lists/tuples return a list of telescopes
    with one shared reference spectrum. position_arcsec defaults to field centre;
    for a list it may also be (n_models, 2), one position per model.
    A non-None detector overrides the existing detector.
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


def spectral_fisher(models, *, noise="poisson", variance=1., variance_fn=None,
                    bad_masks=None):
    """Return dense bin-flux Fisher = sum J.T diag(1 / variance) J.

    Gaussian: variance is a fixed scalar or image-shaped array (variance, not
    sigma); for multiple models it may be a list with one entry per model.
    Poisson: variance is the reference prediction, strictly positive
    on included pixels. variance_fn(mu) overrides either noise choice and is
    frozen at the reference. bad_masks=True excludes a pixel; for multiple
    models pass a list containing one mask (or None) per model.

    Monochromatic optical responses are computed one at a time. The detector
    is differentiated at the full polychromatic reference using JVP, so a
    differentiable nonlinear detector is handled locally too.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before building models")
    if noise not in ("gaussian", "poisson"):
        raise ValueError("noise must be 'gaussian' or 'poisson'")
    models = list(models) if isinstance(models, (list, tuple)) else [models]
    if not models or not all(isinstance(m.source, SpectralSource) for m in models):
        raise ValueError("Use attach_spectrum first")
    masks = [None] * len(models) if bad_masks is None else (
        [bad_masks] if len(models) == 1 else list(bad_masks))
    if len(masks) != len(models):
        raise ValueError("Supply one bad mask per model")
    variances = list(variance) if isinstance(variance, (list, tuple)) else [variance] * len(models)
    if len(variances) != len(models):
        raise ValueError("Supply one variance per model")
    reference = models[0].source
    n = reference.bin_flux.size
    F = np.zeros((n, n))
    for model, bad, model_variance in zip(models, masks, variances):
        source, optics = model.source, model.optics
        if not onp.array_equal(source.wavelengths, reference.wavelengths) or not onp.array_equal(source.bin_flux, reference.bin_flux):
            raise ValueError("Models must share wavelengths and reference bin fluxes")
        mono = jax.jit(lambda wavelength: source.monochromatic_psf(optics, wavelength))
        psfs = [mono(w) for w in source.wavelengths]
        responses = np.stack([p.data for p in psfs]) * source.filt_weights[:, None, None]
        optical = np.tensordot(source.bin_flux, responses, axes=1)
        template = psfs[0]

        def detect(data):
            return data if model.detector is None else model.detector.model(template.set("data", data))

        mu, pushforward = jax.linearize(detect, optical)
        if mu.ndim != 2:
            raise ValueError("The detector must return a 2D image array")
        keep = onp.ones(mu.shape, dtype=bool) if bad is None else ~onp.asarray(bad, dtype=bool)
        if keep.shape != mu.shape or not keep.any():
            raise ValueError("Bad mask must match the output image and retain pixels")
        idx = np.asarray(onp.flatnonzero(keep))
        selected_mu = onp.asarray(mu).ravel()[onp.asarray(idx)]
        if not onp.isfinite(selected_mu).all() or (noise == "poisson" and onp.any(selected_mu < 0)):
            raise ValueError("Invalid reference image on included pixels")
        var = variance_fn(mu) if variance_fn is not None else (mu if noise == "poisson" else model_variance)
        var = np.broadcast_to(np.asarray(var), mu.shape).reshape(-1)[idx]
        if not onp.all(onp.isfinite(var) & (onp.asarray(var) > 0)):
            raise ValueError("Included variances must be finite and positive; mask zero-mean Poisson pixels explicitly")
        column = jax.jit(lambda tangent: pushforward(tangent).reshape(-1)[idx])
        J = np.stack([column(response) for response in responses], axis=1)
        F = F + J.T @ (J / var[:, None])
    return (F + F.T) / 2


@dataclass
class SpectralModes:
    wavelengths: jax.Array
    filt_weights: jax.Array
    reference: jax.Array
    fisher: jax.Array
    values: jax.Array
    vectors: jax.Array

    @property
    def sigma(self):
        """Conditional 1-sigma errors in incident-bin-photon modal coordinates."""
        positive = self.values > 0
        return np.where(positive, 1 / np.sqrt(np.where(positive, self.values, 1.)), np.inf)

    def coefficients(self, bin_flux):
        """Project a spectrum onto retained columns; useful as basis_weights."""
        return self.vectors.T @ np.asarray(bin_flux)


def leading_modes(fisher, k=6):
    """Return k largest eigenvalues and unit-norm column eigenvectors.

    No flux/DC direction is removed. Asking for k modes does not establish
    that all k are measured significantly. k can equal the number of bins.
    """
    fisher = np.asarray(fisher)
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError("fisher must be square")
    if not isinstance(k, (int, onp.integer)) or not 1 <= k <= fisher.shape[0]:
        raise ValueError("Require 1 <= k <= number of wavelength bins")
    values, vectors = np.linalg.eigh((fisher + fisher.T) / 2)
    values, vectors = values[::-1], vectors[:, ::-1]
    scale = max(float(np.max(np.abs(values))), onp.finfo(float).tiny)
    if float(values[-1]) < -1e-9 * scale:
        raise ValueError("Significantly negative Fisher eigenvalue")
    # Remove numerical null eigenvalues, including small positive roundoff.
    values = np.where(values > 10 * fisher.shape[0] * onp.finfo(float).eps * scale, values, 0.)
    vectors = vectors[:, :k]
    pivots = np.argmax(np.abs(vectors), axis=0)
    signs = np.where(vectors[pivots, np.arange(k)] < 0, -1., 1.)
    return values[:k], vectors * signs


def spectral_eigenmodes(model_or_optics, spectrum_args, k=6, *, detector=None,
                       position_arcsec=(0., 0.), **noise_options):
    """One-call injection, Fisher calculation and leading spectral modes."""
    models = attach_spectrum(model_or_optics, spectrum_args, detector=detector,
                             position_arcsec=position_arcsec)
    fisher = spectral_fisher(models, **noise_options)
    values, vectors = leading_modes(fisher, k)
    source = (models[0] if isinstance(models, list) else models).source
    return SpectralModes(source.wavelengths, source.filt_weights, source.bin_flux,
                         fisher, values, vectors)


def mode_criterion(modes):
    """Fractional modal uncertainty 1 / (|a0| sqrt(d_i)) for every computed mode.

    d_i are the Fisher eigenvalues (modes.values) and a0 = v_0 . q0 is the
    reference spectrum's coefficient on the LEADING mode, both in the
    incident-photons-per-bin coordinates of spectral_fisher. The value scales
    as N**-1/2 with the detected photon count N = reference image sum, so
    build the Fisher at the real flux. Null modes (d_i = 0) give inf.
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
    dluxtras.spectra.recommend_wavelength_sampling (n_coarse ~ 2K).
    """
    if not 0 < threshold:
        raise ValueError("threshold must be positive")
    passed = mode_criterion(modes) < threshold
    count = int(passed.sum())
    if count == passed.size:
        warnings.warn("All computed modes are constrainable: K is a lower bound; "
                      "compute more modes or wavelength bins.", stacklevel=2)
    return count
