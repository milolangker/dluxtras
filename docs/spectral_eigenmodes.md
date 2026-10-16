# Spectral eigenmodes

How many independent numbers about a source's spectrum can an image actually measure? The
spectral Fisher matrix answers this. Its eigenvectors are the spectral shapes the data
constrain, ordered from best to worst, and **K**, the number of *constrainable* modes, sets
how finely you need to sample wavelengths (see
[Spectra](spectra.md#how-many-wavelengths)).

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)

from dluxtras.throughput import spectrum_args
from dluxtras.spectral_eigenmodes import spectral_eigenmodes, constrainable_modes, mode_criterion

args = spectrum_args(wavelengths_m, throughput, nwavels=100, photons=1e9)
modes = spectral_eigenmodes(telescope, args, k=100, noise="poisson")
K = constrainable_modes(modes)       # modes with 1 / (|a0| sqrt(d_i)) < 0.1
criterion = mode_criterion(modes)    # the value for every mode
```

`telescope` can be a `dl.Telescope`, bare optics (a telescope is created; optionally pass
`detector=`), or a list of models sharing the spectrum, e.g. several dithers.

## What is computed

- **Coordinates.** The Fisher matrix is over **incident photons per wavelength bin** (before
  throughput). Every other parameter is held fixed unless you
  [marginalise](#marginalising-nuisance-parameters) it.
- **Construction.** It's `Σ Jᵀ diag(1/variance) J`, built from first derivatives only (no
  Hessian), with one monochromatic response computed at a time.
- **Noise model.** `noise="poisson"` uses the reference image as the variance, which needs
  predictions in counts. `noise="gaussian"` takes a fixed `variance` (a scalar or an
  image-shaped array; a variance, not a standard deviation). `variance_fn=lambda mu: ...`
  overrides both, e.g. `mu + read_noise**2`.
- **Excluded pixels.** `bad_masks=` excludes pixels. Pass one mask for every model, or a
  list with one mask (or `None`) per model; `variance` works the same way.
- **Detector.** It's differentiated at the full polychromatic reference, so a differentiable
  nonlinear detector is handled locally.

`spectral_eigenmodes` does it all in one call. The steps are available separately:
`attach_spectrum` (put a `SpectralSource` on the model), `spectral_fisher` (the matrix), and
`leading_modes` (its k largest eigenpairs, as unit-norm columns).

The result, `SpectralModes`, holds:
- `values`: the eigenvalues `d_i`.
- `vectors`: unit-norm columns.
- `residuals`: eigen-residuals (~1e-15 here; the matrix is dense).
- `sigma`: 1σ per mode.
- `coefficients(bin_flux)`: projects a spectrum onto the modes, e.g. to use as
  `basis_weights`.

These fields match `WavefrontModes` from [wavefront eigenmodes](wavefront_eigenmodes.md),
and the two modules take the same keyword arguments.

## Marginalising nuisance parameters

By default the position, pixel scale, wavefront and so on are treated as perfectly known.
In a real fit some of them are fitted too, which costs spectral information. List them to
marginalise them, i.e. to compute the information left once they are fitted freely:

```python
modes = spectral_eigenmodes(optics, args, k=100,
                            marginalise=["source.position"],
                            marginalise_shared=["optics.psf_pixel_scale"])
```

- **`marginalise=[...]`** nuisances are per model: each model gets its own copy (e.g. each
  dither's position).
- **`marginalise_shared=[...]`** nuisances are one copy shared by all models (e.g. the
  pixel scale).
- **Each entry** is a model key, or an array of directions in bin-flux space. For example,
  the reference spectrum itself as a direction marginalises the overall flux scale.

This is exact (a Schur complement of the joint Fisher matrix), works with mixed units, and
uses the same machinery as [wavefront eigenmodes](wavefront_eigenmodes.md#marginalising-nuisance-parameters).

### Example: does an unknown pixel scale matter?

PSF size scales with λ / pixel scale, so an unknown pixel scale should be partly degenerate
with the spectrum's colour. Here's a 1 m circular aperture, 48 × 48 px at 0.06″/px
(Nyquist at 600 nm), 100 bins and 1e10 incident photons, with a ~20% band (550–650 nm):

| marginalised | K | criterion, modes 1–6 |
|---|---|---|
| nothing | 11 | 1.1e-5, 8.3e-5, 1.7e-4, 2.6e-4, 3.8e-4, 6.5e-4 |
| source position | 11 | identical to 3 digits |
| pixel scale | 11 | 1.1e-5, **1.6e-4**, 1.8e-4, 3.8e-4, 4.2e-4, 1.1e-3 |
| position + pixel scale | 10 | 1.1e-5, 1.6e-4, 1.8e-4, 3.8e-4, 4.4e-4, 1.2e-3 |

A ~40% band (480–720 nm) behaved the same way, with K going from 21 to 20.

- **The pixel scale mainly costs the second mode,** the spectral slope or colour. Its
  uncertainty roughly doubles, and later modes lose 7–70%.
- **The first mode is unaffected.** That's the total flux.
- **Position marginalisation costs nothing:** a shift is orthogonal to a colour change.
- **K drops by at most one,** because one marginalised parameter removes roughly one
  direction's worth of information. So marginalise the pixel scale whenever it is fitted or
  uncertain; it matters most for the low-order colour terms.

## The constrainable-mode criterion

Mode `i` is **constrainable** if its fractional uncertainty is below 10%:

$$
\frac{1}{|a_0|\sqrt{d_i}} < 0.1
$$

- `d_i` is the Fisher eigenvalue.
- `a_0 = v_0 · q_0` is the reference spectrum's coefficient on the *leading* mode.

K is the number of modes that pass; `constrainable_modes(modes, threshold=0.1)`.

- **Compute all modes.** Use `k` equal to the number of wavelength bins. If every computed mode
  passes, K is only a lower bound and a warning is issued.
- **Sample finely enough.** Use ≳ 100 bins for wide bands; K is not converged at ~30 bins.
- **Use the real flux.** The criterion scales as N^(-1/2) with the detected photon count N
  (the reference image sum), so build the Fisher matrix at the real flux. Null modes
  (`d_i = 0`) give infinity.
- **Mind the metric.** The eigenvectors are unit-norm in incident-photons-per-bin coordinates,
  a metric that depends on the wavelength sampling; no bin-width whitening is applied.

Then use K to choose wavelength sampling:
`recommend_wavelength_sampling(K, photons=N)` suggests `n_coarse ≈ 2K`.
