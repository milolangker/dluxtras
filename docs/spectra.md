# Spectra, sources and throughput

These modules describe a broadband source as seen through an instrument: a source spectrum,
times the system throughput, propagated through dLux optics. They can propagate far fewer
wavelengths than the spectrum has bins, using **Hermite wavelength sparsity**.

## Conventions

- **Wavelengths** are in metres everywhere. `load_throughput` reads other units and converts.
- **Throughput** is dimensionless and *excludes* the source spectrum: optics × filter ×
  detector QE, as appropriate for your bench.
- **Incident vs detected photons.** `spec_weights()` is the *incident* photons per bin (before
  throughput), and `filt_weights` the mean throughput per bin. So:
    - `spectrum.flux` = incident photons = `sum(spec_weights())`
    - `spectrum.proper_flux()` = *detected* photons = `sum(spec_weights() * filt_weights)`
- **`spectrum_args(..., photons=)`** sets the *incident* count, split evenly over the bins.
- **Positions:** `PointSourceArcsec` takes arcsec. A plain `dl.PointSource`, and
  `HermitePointSource`, take radians.

## Building a spectrum

`spectrum_args` turns sampled wavelengths and throughput into the arguments of a
`CombinedBasisSpectrum`. It bins the throughput into `nwavels` bins of mean throughput,
trims the band where throughput is below `threshold` (default 1%) of its peak, and by default
gives every bin an independent photon count (an identity basis):

```python
from dluxtras import CombinedBasisSpectrum, spectrum_args, load_throughput

table = load_throughput("bench_throughput.dat", wavelength_unit="um")  # (n, 2): [m, throughput]
args = spectrum_args(table[:, 0], table[:, 1], nwavels=100, photons=1e6)
spectrum = CombinedBasisSpectrum(**args)
spectrum.proper_flux()   # detected photons
```

Pass `basis` and `basis_weights` to describe the spectrum with fewer coefficients, e.g. the
spectral eigenmodes from [Spectral eigenmodes](spectral_eigenmodes.md). `proj` applies a
further linear map to the coefficients. Use `CombinedSpectrum` directly for a subclass with
its own `spec_weights()`.

Other throughput helpers:

- `bin_throughput` returns bin centres and mean throughput.
- `fractional_bandwidth` returns Δλ/λ_c and the central wavelength.
- `filter_integrate` and `trim_throughput` are the building blocks of the above.

## Sources

Use the sources in `dluxtras.sources` rather than `dl.PointSource` with these spectra:

- **`HermitePointSource`** behaves like `dl.PointSource`, but honours the spectrum's
  `n_coarse` and `low_memory`. With neither set, it's identical to `dl.PointSource`.
- **`PointSourceArcsec`** additionally takes its position in arcsec, and sets its flux to the
  spectrum's *detected* flux, `spectrum.proper_flux()`. So the image sums to about the
  detected photon count, minus light falling outside the stamp.

```python
import dLux as dl
from dluxtras import PointSourceArcsec

telescope = dl.Telescope(optics, PointSourceArcsec(spectrum=spectrum, position=np.zeros(2)))
image = telescope.model()
```

## Hermite wavelength sparsity

A broadband PSF is a weighted sum of monochromatic PSFs, one per wavelength bin. Within a band
the PSF varies smoothly with wavelength, so it's enough to propagate a few **coarse nodes**,
each with its derivative with respect to wavelength (one forward-mode pass gives both), and
cubic-Hermite-interpolate the rest.

```python
spectrum = CombinedBasisSpectrum(**spectrum_args(wl, tp, nwavels=100, photons=1e6, n_coarse=12))
spectrum.coarse_wavelengths   # the 12 wavelengths actually propagated
```

- **`n_coarse=None`** (the default) propagates every wavelength.
- **An integer** propagates that many evenly spread nodes, both ends included.
- **`low_memory=True`** propagates nodes one at a time, recomputing them in the backward pass.
  Peak memory is about that of one node, and gradients are ~1.5–2× slower. It works with or
  without `n_coarse`.

### How many wavelengths?

`recommend_wavelength_sampling(K, photons)` gives a recommendation from the number of
constrainable spectral modes `K` (see [Spectral eigenmodes](spectral_eigenmodes.md)) and the
detected photon count `N`. Its empirical rules come from a point source with Poisson noise,
requiring the model error to stay below Δχ² < 1, tested across 4 filters, 32–256 px stamps and
N = 1e8–1e10. Neither rule ever undershot, there or in a [generic re-check](wavelength_sampling.md).

- **Hermite:** `n_coarse = max(3, 2K)`, with `n_fine = max(300, 20K)` fine wavelengths. Fine
  wavelengths are almost free with Hermite.
- **Exact:** `n_exact = ceil(23.5 K^0.553 (N/1e9)^0.208)`.

One Hermite node costs about `jvp_cost` (default 2.5) exact wavelengths. If the estimated
speedup `n_exact / (jvp_cost · n_coarse)` is below `min_speedup` (default 1.5), it recommends
plain propagation of `n_exact` wavelengths instead.

```python
from dluxtras import recommend_wavelength_sampling

rec = recommend_wavelength_sampling(K=6, photons=1e9)
rec["n_coarse"], rec["n_fine"], rec["sparse"], rec["speedup"]
```

## Low-level pieces

`propagate_spectrum(optics, spectrum, offset, flux)` is what the sources call. It's
equivalent to `optics.propagate(spectrum.wavelengths, offset, spectrum.weights * flux)`, but
honours `n_coarse` and `low_memory`. `coarse_indices` and `hermite_node_coeffs` give the
node positions and the Hermite interpolation weights.
