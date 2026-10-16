# dluxtras

Extensions for [∂Lux](https://github.com/LouisDesdoigts/dLux): reusable pieces shared by several
dLux-based projects.

!!! note "AI disclosure"
    Most of the code in this repository, and its documentation, was either written or rewritten
    with [Claude](https://claude.com/claude-code) (Anthropic's AI assistant), under the direction
    and review of the repository owner. Treat it with the usual scrutiny, and check results that
    matter.

## What's in it

| Module | Contents | Guide |
| --- | --- | --- |
| `dluxtras.spectra` | `CombinedSpectrum` / `CombinedBasisSpectrum` (a source spectrum times a system throughput), `propagate_spectrum` with **Hermite wavelength sparsity**, and `recommend_wavelength_sampling`. | [Spectra, sources and throughput](spectra.md) |
| `dluxtras.sources` | `HermitePointSource` (honours Hermite sparsity) and `PointSourceArcsec` (position in arcsec, flux in detected photons). | [Spectra, sources and throughput](spectra.md) |
| `dluxtras.throughput` | Loading and binning system throughput; `spectrum_args` builds a `CombinedBasisSpectrum`. | [Spectra, sources and throughput](spectra.md) |
| `dluxtras.spectral_eigenmodes` | Spectral Fisher matrix, its eigenmodes, and the number of constrainable modes K, optionally marginalising any model parameter. | [Spectral eigenmodes](spectral_eigenmodes.md) |
| `dluxtras.wavefront_eigenmodes` | `MaskedPixelOPD` (one OPD value per pupil pixel) and matrix-free wavefront Fisher eigenmodes, marginalising positions, pixel scale, focus or any other parameter. | [Wavefront eigenmodes](wavefront_eigenmodes.md) |
| `dluxtras.eigenmodes` | The shared core: Fisher operators, marginalisation and `leading_modes`. | [Wavefront eigenmodes](wavefront_eigenmodes.md#marginalising-nuisance-parameters) |
| `dluxtras.fitting` | `FitProblem`, `Curvature` and `FisherFit`: Fisher-whitened fitting with SGD, Nesterov momentum, L-BFGS or natural gradient, plus uncertainties. | [Fitting](fitting.md) |
| `dluxtras.curvature` | Unit-free (Jacobi-normalised) Hessians, covariances and correlations. | [Unit-free Hessians](curvature.md) |

## Install

```bash
pip install "dluxtras @ git+https://github.com/milolangker/dluxtras.git"
# or, for development
git clone https://github.com/milolangker/dluxtras.git && pip install -e "./dluxtras[test]"
```

It needs dLux 0.15.x. Enable double precision before building models; the Fisher and fitting
tools require it:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Quick example

Build a broadband point source whose 100-bin spectrum is propagated at only 12 wavelengths, then
fit its position:

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import dLux as dl
from dluxtras import CombinedBasisSpectrum, FitProblem, PointSourceArcsec, gaussian_nll, spectrum_args

wavelengths = np.linspace(500e-9, 700e-9, 201)  # m
throughput = np.exp(-0.5 * ((wavelengths - 600e-9) / 40e-9) ** 2)
spectrum = CombinedBasisSpectrum(**spectrum_args(wavelengths, throughput, nwavels=100,
                                                 photons=1e6, n_coarse=12))

optics = dl.AngularOpticalSystem(256, 0.1, [dl.CircularAperture(0.05, normalise=True)], 64, 0.1, 2)
telescope = dl.Telescope(optics, PointSourceArcsec(spectrum=spectrum, position=np.zeros(2)))
data = telescope.set("source.position", np.array([0.02, -0.01])).model()

problem = FitProblem(telescope, {"source.position": np.zeros(2)}, data, gaussian_nll,
                     np.sqrt(data + 1.0))
fit = problem.fit(optimiser="lbfgs").run(50)
print(fit.params["source.position"], problem.curvature(fit.params, precondition="full").std)
```

## Building these docs

```bash
pip install -e ".[docs]"
zensical serve           # live preview at http://localhost:8000
zensical build --strict  # what CI runs
```

## Credits

`CombinedSpectrum` and `CombinedBasisSpectrum` are adapted from
[h-greer/hubble](https://github.com/h-greer/hubble) (MIT License, © 2024 h-greer). The Hermite
sparsity, `proj`, `low_memory`, the incident/detected flux conventions and the spectral Fisher
tools were added on top. dluxtras is released under the MIT License.
