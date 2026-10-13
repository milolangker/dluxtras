"""Point sources that honour Hermite wavelength sparsity (spectrum.n_coarse)."""

import dLux as dl
import dLux.utils as dlu

from .spectra import propagate_spectrum


class HermitePointSource(dl.PointSource):
    """dl.PointSource that honours spectrum.n_coarse (Hermite wavelength sparsity).

    Identical to dl.PointSource when the spectrum has no n_coarse (or it is None).
    """

    def model(self, optics, return_wf=False, return_psf=False):
        source = self.normalise()
        return propagate_spectrum(
            optics, source.spectrum, source.position, source.flux,
            return_wf=return_wf, return_psf=return_psf,
        )


class PointSourceArcsec(HermitePointSource):
    """Position in arcseconds; flux is the DETECTED flux spectrum.proper_flux()."""

    def model(self, optics, return_wf=False, return_psf=False):
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
