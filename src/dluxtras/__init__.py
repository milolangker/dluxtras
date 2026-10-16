"""dluxtras: extensions for dLux (spectra, sources, eigenmodes, fitting, ...)."""

__version__ = "0.1.0"

from . import (
    curvature,
    eigenmodes,
    fitting,
    sources,
    spectra,
    spectral_bins,
    spectral_eigenmodes,
    throughput,
    wavefront_eigenmodes,
)
from .curvature import correlation, scaled_covariance, scaled_hessian, unscale
from .eigenmodes import FisherOperator, leading_modes
from .fitting import Curvature, FisherFit, FitProblem, gaussian_nll
from .sources import HermitePointSource, PointSourceArcsec
from .spectra import (
    CombinedBasisSpectrum,
    CombinedSpectrum,
    propagate_spectrum,
    recommend_wavelength_sampling,
)
from .spectral_bins import SpectralBins, optimal_bins
from .spectral_eigenmodes import constrainable_modes, spectral_eigenmodes, spectral_fisher
from .throughput import bin_throughput, load_throughput, spectrum_args
from .wavefront_eigenmodes import (
    MaskedPixelOPD,
    nuisance_shapes,
    wavefront_eigenmodes,
    wavefront_fisher,
)

__all__ = [
    "CombinedSpectrum",
    "CombinedBasisSpectrum",
    "propagate_spectrum",
    "recommend_wavelength_sampling",
    "HermitePointSource",
    "PointSourceArcsec",
    "spectral_fisher",
    "spectral_eigenmodes",
    "constrainable_modes",
    "optimal_bins",
    "SpectralBins",
    "MaskedPixelOPD",
    "nuisance_shapes",
    "wavefront_fisher",
    "wavefront_eigenmodes",
    "FisherOperator",
    "leading_modes",
    "load_throughput",
    "bin_throughput",
    "spectrum_args",
    "scaled_hessian",
    "unscale",
    "scaled_covariance",
    "correlation",
    "FitProblem",
    "Curvature",
    "FisherFit",
    "gaussian_nll",
]
