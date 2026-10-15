"""dluxtras: extensions for dLux (spectra, sources, eigenbases, ...)."""

__version__ = "0.1.0"

from . import curvature, fitting, sources, spectra, spectral_eigenmodes, throughput
from .curvature import correlation, scaled_covariance, scaled_hessian, unscale
from .fitting import Curvature, FisherFit, FitProblem, gaussian_nll
from .sources import HermitePointSource, PointSourceArcsec
from .spectra import (
    CombinedBasisSpectrum,
    CombinedSpectrum,
    propagate_spectrum,
    recommend_wavelength_sampling,
)
from .spectral_eigenmodes import constrainable_modes, spectral_eigenmodes, spectral_fisher
from .throughput import bin_throughput, load_throughput, spectrum_args

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
