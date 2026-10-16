"""Point sources that honour Hermite wavelength sparsity.

The sources here route propagation through `dluxtras.spectra.propagate_spectrum`, so a
spectrum's `n_coarse` (Hermite wavelength sparsity) and `low_memory` settings are respected.
"""

import dLux as dl
import dLux.utils as dlu

from .spectra import propagate_spectrum


class HermitePointSource(dl.PointSource):
    """`dl.PointSource` that honours `spectrum.n_coarse` (Hermite wavelength sparsity).

    Gives the same images as `dl.PointSource` when the spectrum has no `n_coarse` (or it
    is None) and `low_memory` is off. It does not run dLux's validation of the
    `return_wf` / `return_psf` options; `return_wf` is not supported with `n_coarse` or
    `low_memory`.
    Propagation goes through `dluxtras.spectra.propagate_spectrum`, which also honours
    `spectrum.low_memory`. The constructor is inherited unchanged from `dl.PointSource`,
    and takes these attributes as keyword arguments.

    Attributes
    ----------
    wavelengths : Array, optional
        Wavelengths in metres at which the spectrum is defined. Ignored if `spectrum`
        is given. Default None.
    position : Array, optional
        (x, y) on-sky position in radians. Default (0, 0).
    flux : float, optional
        Flux in photons; multiplies the normalised spectral weights. Default 1.0.
    weights : Array, optional
        Spectral weights, used with `wavelengths` when `spectrum` is not given.
        Default None.
    spectrum : dl.spectra.BaseSpectrum, optional
        Spectrum of the source, e.g. a `dluxtras.spectra.CombinedSpectrum`. Default None.
    """

    def model(self, optics, return_wf=False, return_psf=False):
        """Propagate the normalised source through `optics`.

        The source is normalised (`self.normalise()`) and then propagated with
        `propagate_spectrum(optics, spectrum, position, flux, ...)`, i.e. with weights
        `spectrum.weights * flux`.

        Parameters
        ----------
        optics : dl.BaseOpticalSystem
            Optical system to propagate through.
        return_wf : bool, optional
            Return the wavefront instead of an image. Not supported when the spectrum
            uses `n_coarse` or `low_memory`. Default False.
        return_psf : bool, optional
            Return a `dl.PSF` object instead of an image array. Default False.

        Returns
        -------
        Array or dl.PSF or Wavefront
            The polychromatic image (or `dl.PSF`, or wavefront when supported).

        Raises
        ------
        ValueError
            If `return_wf` is True for a Hermite-interpolated or low-memory spectrum.
        """
        source = self.normalise()
        return propagate_spectrum(
            optics, source.spectrum, source.position, source.flux,
            return_wf=return_wf, return_psf=return_psf,
        )


class PointSourceArcsec(HermitePointSource):
    """`HermitePointSource` with position in arcseconds and flux taken from the spectrum.

    At model time the position is converted from arcseconds to radians and the flux is
    replaced by the DETECTED flux `spectrum.proper_flux()`; the stored `flux` attribute is
    not used. The spectrum must therefore provide `proper_flux()` (e.g. a
    `dluxtras.spectra.CombinedSpectrum`). The constructor is inherited unchanged from
    `dl.PointSource`.

    Attributes
    ----------
    wavelengths : Array, optional
        Wavelengths in metres at which the spectrum is defined. Ignored if `spectrum`
        is given. Default None.
    position : Array, optional
        (x, y) on-sky position in arcseconds. Default (0, 0).
    flux : float, optional
        Stored for compatibility with `dl.PointSource`; replaced by
        `spectrum.proper_flux()` in `model`. Default 1.0.
    weights : Array, optional
        Spectral weights, used with `wavelengths` when `spectrum` is not given.
        Default None.
    spectrum : dl.spectra.BaseSpectrum, optional
        Spectrum providing `proper_flux()`, e.g. a `dluxtras.spectra.CombinedSpectrum`.
        Default None.
    """

    def model(self, optics, return_wf=False, return_psf=False):
        """Propagate through `optics` with arcsecond position and detected flux.

        Sets `position` to `arcsec2rad(position)` and `flux` to `spectrum.proper_flux()`,
        then calls `HermitePointSource.model`.

        Parameters
        ----------
        optics : dl.BaseOpticalSystem
            Optical system to propagate through.
        return_wf : bool, optional
            Return the wavefront instead of an image. Not supported when the spectrum
            uses `n_coarse` or `low_memory`. Default False.
        return_psf : bool, optional
            Return a `dl.PSF` object instead of an image array. Default False.

        Returns
        -------
        Array or dl.PSF or Wavefront
            The polychromatic image (or `dl.PSF`, or wavefront when supported).

        Raises
        ------
        ValueError
            If `return_wf` is True for a Hermite-interpolated or low-memory spectrum.
        """
        source = self.set(
            ["position", "flux"],
            [
                dlu.arcsec2rad(self.position),
                self.spectrum.proper_flux(),
            ],
        )
        return HermitePointSource.model(
            source, optics,
            return_wf=return_wf,
            return_psf=return_psf,
        )
