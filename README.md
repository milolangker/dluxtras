> **AI disclosure.** Most of the code in this repository, and its documentation, was either written or rewritten with [Claude](https://claude.com/claude-code) (Anthropic's AI assistant), under the direction and review of the repository owner. Treat it with the usual scrutiny, and check results that matter.

# dluxtras

Extensions for [∂Lux](https://github.com/LouisDesdoigts/dLux): reusable pieces shared by several dLux-based projects. It currently holds the spectral tools:

| Module | Contents |
| --- | --- |
| `dluxtras.spectra` | `CombinedSpectrum` / `CombinedBasisSpectrum`: a source spectrum (`spec_weights`, incident photons per bin) times a system throughput (`filt_weights`). Also `propagate_spectrum`, with **Hermite wavelength sparsity**: set `n_coarse` to propagate only that many wavelengths, each with its derivative, and cubic-Hermite-interpolate the rest. And `recommend_wavelength_sampling`, which gives `n_coarse ≈ 2K` from the number of constrainable modes K. |
| `dluxtras.sources` | `HermitePointSource` (a `dl.PointSource` that honours `n_coarse`) and `PointSourceArcsec` (position in arcsec, flux = detected photons). |
| `dluxtras.throughput` | Loading and binning system throughput, and `spectrum_args(...)`, which builds the arguments for a `CombinedBasisSpectrum`. |
| `dluxtras.spectral_eigenmodes` | Spectral Fisher matrix and eigenmodes, `mode_criterion` and `constrainable_modes` (K). |

More will follow, e.g. other eigenbases and Fisher-based fitting.

## Install

```bash
pip install "dluxtras @ git+https://github.com/milolangker/dluxtras.git"
# or, for development
git clone https://github.com/milolangker/dluxtras.git && pip install -e "./dluxtras[test]"
```

It needs dLux 0.15.x. Enable double precision before building models, with `jax.config.update("jax_enable_x64", True)`. The Fisher tools require it.

## Example

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import dLux as dl
from dluxtras import CombinedBasisSpectrum, PointSourceArcsec, spectrum_args

wavelengths = np.linspace(500e-9, 700e-9, 201)  # m
throughput = np.exp(-0.5 * ((wavelengths - 600e-9) / 40e-9) ** 2)
spectrum = CombinedBasisSpectrum(**spectrum_args(wavelengths, throughput, nwavels=100,
                                                 photons=1e6, n_coarse=12))

optics = dl.AngularOpticalSystem(256, 0.1, [dl.CircularAperture(0.05, normalise=True)], 64, 0.1, 2)
telescope = dl.Telescope(optics, PointSourceArcsec(spectrum=spectrum, position=np.zeros(2)))
image = telescope.model()  # sums to ~spectrum.proper_flux(); 12 propagated wavelengths stand in for 100
```

## Fitting

`FitProblem` holds what is fitted (model, parameters, data, likelihood); its `curvature()` is the
Fisher matrix in unit-free form; `fit()` runs a resumable optimiser in Fisher-whitened coordinates.

```python
from dluxtras import FitProblem, gaussian_nll

params = {"source.position": np.zeros(2), "aberrations.coefficients": np.zeros(8)}
problem = FitProblem(telescope, params, data, gaussian_nll, std)
fit = problem.fit()                       # default: 1/diag(Fisher) preconditioned SGD, lr 1
fit.run(200).refresh().run(200)           # refresh: recompute the Fisher where the fit is now
fit = problem.fit(precondition="full", optimiser="lbfgs").run(50)   # usually much faster
sigma = problem.curvature(fit.params, precondition="full").std   # marginal 1-sigma
```

The likelihood is any `nll(prediction, data, *args)`; the Fisher is exact for Gaussian noise of
fixed (co)variance, correlated or not, and for Poisson noise. See `dluxtras.fitting` for details, and `benchmarks/fitting_benchmark.md` for which method to use.

## Tests

```bash
pytest
```

## Credits

`CombinedSpectrum` and `CombinedBasisSpectrum` are adapted from [h-greer/hubble](https://github.com/h-greer/hubble) (MIT License, © 2024 h-greer); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The Hermite sparsity, `proj`, `low_memory`, the incident/detected flux conventions and the spectral Fisher tools were added on top.

## License

MIT; see [LICENSE](LICENSE).
