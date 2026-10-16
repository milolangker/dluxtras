# Wavefront eigenmodes

Which wavefront shapes can an image actually measure, and how well? The Fisher matrix of a
wavefront parameterisation answers this. Its leading eigenvectors are the OPD shapes the
data constrain best, and `1/sqrt(eigenvalue)` is each shape's 1σ uncertainty.

`dluxtras.wavefront_eigenmodes` computes these **matrix-free**: the Fisher matrix is only
ever applied to vectors, and its leading eigenpairs come from Lanczos iteration. That makes
one free OPD value per pupil pixel (thousands of parameters) practical. Nuisance parameters
(source positions, pixel scale, focus terms, anything in the model) can be
[marginalised](#marginalising-nuisance-parameters).

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)

from dluxtras.wavefront_eigenmodes import MaskedPixelOPD, nuisance_shapes, wavefront_eigenmodes

fit_opd = MaskedPixelOPD(pupil)   # pupil: the aperture's transmission array
models = [m.set("optics", m.optics.insert_layer(("fit_opd", fit_opd), 1)) for m in models]

modes = wavefront_eigenmodes(models, "optics.fit_opd.coefficients", k=6,
                             variance_fn=lambda mu: mu + read_noise**2)
modes.values      # eigenvalues, descending
modes.sigma       # 1-sigma of each mode's coefficient, in nm
modes.maps()      # (6, ny, nx) mode maps, NaN outside the pupil
modes.residuals   # Lanczos residuals; check they are small
```

## The pixel OPD layer

`MaskedPixelOPD(transmission, threshold=0)` adds one OPD coefficient, in **nm**, per
transmitting pupil pixel.

- **It only adds OPD.** Transmission stays with the aperture layer, so insert it in the same
  pupil plane, on the same grid (index 1, right after the aperture, in the example).
- **Pick a fresh layer name,** e.g. `"fit_opd"`, if the optics already contain an OPD layer.
- **`threshold`** excludes faint edge pixels from the fit, but not from the optics.
- **Static OPD can stay.** Any existing static OPD stays in the optical train, so the modes
  are computed about that reference.

The key doesn't have to be a pixel OPD, though. `wavefront_eigenmodes(models, key)` works for
any parameter vector of the model, e.g. `"optics.aberrations.coefficients"` of a Zernike
`BasisOptic`.

## Several exposures

Pass a list of models (e.g. dithers, rolls, defocused images). They share **one** wavefront:
the parameter at `key` is set to the same `q0` in every model (by default, its value in the
first). Each model's own flux must already include its integration time. A single model
with a stacked `(n_exposures, ny, nx)` output also works.

Noise options are the same as for [spectral eigenmodes](spectral_eigenmodes.md):

- **`noise="poisson"`** (the default) uses the reference image as the variance, so it must
  be in counts.
- **`noise="gaussian"`** takes a fixed `variance`.
- **`variance_fn=lambda mu: ...`** overrides both.
- **`bad_masks`** excludes pixels.
- **Per model, or for all:** `variance` and `bad_masks` can be a list with one entry per
  model, or a single value used for all of them.

## Marginalising nuisance parameters

By default every other model parameter is held fixed at its reference value. That's
optimistic when, in a real fit, you'd also fit the source positions or a focus term: the
wavefront then has to share information with them. Listing them marginalises them, i.e.
computes the information about the wavefront left once those nuisances are fitted freely.

There are two lists:

- **`marginalise=[...]`** nuisances are **per model**: each exposure gets its own
  independent copy. Use it for things that change between exposures, such as source
  positions.
- **`marginalise_shared=[...]`** nuisances are **shared**: one copy for all models. Use it
  for properties of the instrument, such as the pixel scale.

Each entry is either a **model key** or an **array of directions** in the parameter's
space. Mix them freely:

```python
modes = wavefront_eigenmodes(
    models, "optics.fit_opd.coefficients", k=10,
    marginalise=["source.position", nuisance_shapes(fit_opd, ["defocus"])],
    marginalise_shared=["optics.psf_pixel_scale"],
)
```

- **A key** (e.g. `"source.position"`, `"source.flux"`, `"optics.psf_pixel_scale"`, or a
  `BasisOptic`'s coefficients) marginalises every entry of that parameter.
- **Directions** marginalise those shapes' coefficients. `nuisance_shapes(layer, terms)`
  gives orthonormal tip, tilt and defocus shapes over a `MaskedPixelOPD`'s pixels, plus any
  custom shapes you pass. The resulting modes are then orthogonal to them.

All nuisances are marginalised together, exactly, via the Schur complement of the joint
Fisher matrix:

$$
F_\mathrm{marg} = F_{qq} - F_{q\eta}\,F_{\eta\eta}^{+}\,F_{\eta q}
$$

- **Mixed units are safe.** `F_ηη` is pseudo-inverted in unit-free (Jacobi-normalised) form,
  so radians, arcseconds and photons can be mixed.
- **Invisible nuisances are ignored.** Combinations the data can't see at all are dropped
  rather than causing an error.
- **Cost.** Each nuisance costs one extra derivative evaluation per model, done once when
  the operator is built.

!!! tip "Tip/tilt: OPD shapes or source position?"
    A tip/tilt OPD and a source offset move the image the same way, so marginalising either
    removes (almost) the same information. Marginalising both at once is harmless: the
    pseudo-inverse handles the degeneracy.

## Piston and other null modes

OPD piston doesn't change images, so for `MaskedPixelOPD` keys it is projected out by
default (`remove_piston=True`). Other null directions remain possible:

- **Relative pistons** between disconnected pupil segments.
- **Directions at a symmetric, aberration-free reference** that have zero first-order effect
  but second-order effects.

Request only as many modes as are informative, and check `values`.

## Units and normalisation

`vectors[:, i]` has unit Euclidean norm over the active pixels. A modal coefficient `c` (nm)
is the OPD `c * vectors[:, i]`, whose pupil RMS is `|c| / sqrt(N_active)`. For unit-RMS
maps, multiply `maps()` by `sqrt(N_active)` and divide `sigma` by it. Eigenvalues depend
on this convention and on the pupil sampling.

## Lower-level pieces

- **`wavefront_fisher(models, key, ...)`** returns a `FisherOperator`. `F(v)` gives `F @ v`,
  and `F.dense()` builds the matrix for small problems.
- **`leading_modes(F, k, project_out=...)`** (in `dluxtras.eigenmodes`) diagonalises either a
  dense matrix or an operator.
- **`mode_maps(layer, vectors)`** scatters vectors onto the pupil grid.
