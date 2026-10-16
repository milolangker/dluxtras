> **AI disclosure.** Most of the code in this repository, and its documentation, was either written or rewritten with [Claude](https://claude.com/claude-code) (Anthropic's AI assistant), under the direction and review of the repository owner. Treat it with the usual scrutiny, and check results that matter.

# dluxtras

Extensions for [∂Lux](https://github.com/LouisDesdoigts/dLux): reusable pieces shared by several dLux-based projects.

**Documentation: <https://milolangker.github.io/dluxtras/>** (guides on when to use each option, plus the full API reference).

| Module | Contents |
| --- | --- |
| `dluxtras.spectra` | `CombinedSpectrum` / `CombinedBasisSpectrum`: a source spectrum times a system throughput, with **Hermite wavelength sparsity** (`n_coarse`) and `recommend_wavelength_sampling`. |
| `dluxtras.sources` | `HermitePointSource` (honours `n_coarse`) and `PointSourceArcsec` (position in arcsec, flux = detected photons). |
| `dluxtras.throughput` | Loading and binning system throughput; `spectrum_args(...)` builds a `CombinedBasisSpectrum`. |
| `dluxtras.spectral_eigenmodes` | Spectral Fisher matrix and eigenmodes, and the number of constrainable modes K. |
| `dluxtras.wavefront_eigenmodes` | `MaskedPixelOPD` and matrix-free wavefront Fisher eigenmodes. |
| `dluxtras.eigenmodes` | Shared core: marginalising any model parameter (per exposure or shared) and `leading_modes`. |
| `dluxtras.fitting` | `FitProblem` / `Curvature` / `FisherFit`: Fisher-whitened fitting (Nesterov SGD, L-BFGS or natural gradient) and uncertainties. |
| `dluxtras.curvature` | Unit-free (Jacobi-normalised) Hessians, covariances and correlations. |

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

```python
from dluxtras import FitProblem, gaussian_nll

problem = FitProblem(telescope, {"source.position": np.zeros(2)}, data, gaussian_nll, std)
fit = problem.fit().run(200)                         # default: Nesterov SGD, diagonal Fisher
fit = problem.fit(optimiser="lbfgs").run(100)        # usually fastest; stops when converged
fit = problem.fit(precondition="full").run(10)       # natural gradient, near the minimum
sigma = problem.curvature(fit.params, precondition="full").std   # marginal 1-sigma
```

See the [fitting guide](https://milolangker.github.io/dluxtras/fitting/) for when to use each.

## Tests and docs

```bash
pytest
pip install -e ".[docs]" && zensical serve   # docs preview at http://localhost:8000
```

## Credits

`CombinedSpectrum` and `CombinedBasisSpectrum` are adapted from [h-greer/hubble](https://github.com/h-greer/hubble) (MIT License, © 2024 h-greer); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The Hermite sparsity, `proj`, `low_memory`, the incident/detected flux conventions and the spectral Fisher tools were added on top.

## License

MIT; see [LICENSE](LICENSE).
