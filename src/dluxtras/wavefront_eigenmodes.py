"""Matrix-free Fisher eigenmodes of a wavefront (or any parameter vector) of dLux models.

`MaskedPixelOPD` adds one OPD coefficient (nm) per transmitting pupil pixel.
`wavefront_eigenmodes` finds the wavefront shapes the data constrain best, optionally
with nuisance parameters marginalised: any model parameter by key (e.g. the source
position, the pixel scale, a defocus term), and/or OPD shapes such as tip, tilt and
defocus (`nuisance_shapes`). The API matches `dluxtras.spectral_eigenmodes`; the
marginalisation rules are described in `dluxtras.eigenmodes`.

The Fisher matrix is never formed: it is applied as Jacobian-vector and vector-Jacobian
products, and its leading eigenpairs found by Lanczos iteration. Memory is
O(N_active * ncv) for the Lanczos vectors, plus the model's autodiff workspace. SciPy
runs on the CPU; each product transfers its result back from JAX.

Examples
--------
```python
import jax
jax.config.update("jax_enable_x64", True)
from dluxtras.wavefront_eigenmodes import MaskedPixelOPD, nuisance_shapes, wavefront_eigenmodes

fit_opd = MaskedPixelOPD(pupil)  # the transmission array of the aperture layer
models = [m.set("optics", m.optics.insert_layer(("fit_opd", fit_opd), 1)) for m in models]
modes = wavefront_eigenmodes(
    models, "optics.fit_opd.coefficients", k=6,
    marginalise=["source.position"],                 # each exposure's own position
    marginalise_shared=["optics.psf_pixel_scale"],   # one pixel scale for all
    variance_fn=lambda mu: mu + read_noise**2)       # Poisson + read noise, in counts
maps = modes.maps()      # (6, ny, nx), NaN outside the fitted pixels
sigma_nm = modes.sigma   # 1-sigma of each mode's coefficient, in nm
```

Notes
-----
The models describe ONE shared wavefront, set by `key`, seen in several exposures; flux
in each model must already include the intended integration time. Index 1 in
`insert_layer` assumes the first optical layer is the aperture. Use a layer name that
does not clash with an existing one (e.g. `"fit_opd"`).

OPD piston has no effect on images, so it is projected out by default for
`MaskedPixelOPD` keys. Relative pistons between disconnected pupil regions remain, and
further null modes can exist. At a symmetric zero-aberration reference some directions
have zero first-order sensitivity despite second-order image effects.

`vectors[:, i]` has Euclidean norm one across active pixels. For a modal coefficient `c`
in nm the physical OPD is `c * vectors[:, i]` nm, with unweighted pupil RMS
`|c| / sqrt(N_active)`. For unit-RMS maps, multiply the maps by `sqrt(N_active)` and
divide `sigma` by it. Eigenvalues depend on this convention and on pupil sampling; do
not rescale individual pixels if you want modes in the uniform OPD pixel metric.
"""

from dataclasses import dataclass
from typing import Any

import dLux as dl
import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

from .eigenmodes import (
    FisherOperator,
    _assemble,
    _each,
    _Linearisation,
    leading_modes,
    noise_weights,
)

__all__ = ["MaskedPixelOPD", "nuisance_shapes", "wavefront_fisher", "wavefront_eigenmodes",
           "WavefrontModes", "mode_maps", "leading_modes"]


class MaskedPixelOPD(dl.OpticalLayer):
    """One fitted OPD coefficient, in nm, per selected pupil pixel.

    The layer only adds OPD; amplitude transmission stays with the aperture layer, so
    place it in the same pupil plane, on exactly the same grid. Construct it outside JIT.
    Any existing static OPD can stay in the optical train.

    Parameters
    ----------
    transmission : Array
        2D pupil amplitude transmission, on the wavefront grid.
    threshold : float, optional
        Pixels are fitted where `|transmission| > threshold * max|transmission|`. The
        default 0 fits every nonzero pixel, including partially transmitting edges.
        Raising it excludes pixels from the fit, not from optical transmission.
    coefficients : Array, optional
        Starting OPD per active pixel, in nm, shape `(N_active,)`. Default zeros.

    Attributes
    ----------
    coefficients : Array
        OPD per active pixel, in nm.
    indices : Array
        Flat indices of the active pixels in the pupil grid.
    shape : tuple
        Pupil grid shape.

    Raises
    ------
    ValueError
        If `transmission` is not a finite 2D array, `threshold` is not in [0, 1), no pixel
        is selected, or `coefficients` has the wrong length.
    """

    coefficients: jax.Array
    indices: jax.Array
    shape: tuple = eqx.field(static=True)

    def __init__(self, transmission, threshold=0.0, coefficients=None):
        transmission = onp.asarray(transmission)
        if transmission.ndim != 2 or not onp.isfinite(transmission).all():
            raise ValueError("transmission must be a finite 2D array")
        if not 0 <= threshold < 1:
            raise ValueError("threshold must be in [0, 1)")
        amplitude = onp.abs(transmission)
        indices = onp.flatnonzero(amplitude > threshold * amplitude.max())
        if indices.size == 0:
            raise ValueError("No transmitting pixels selected")
        self.shape = transmission.shape
        self.indices = np.asarray(indices)
        self.coefficients = (np.zeros(indices.size) if coefficients is None
                             else np.asarray(coefficients, dtype=float))
        if self.coefficients.shape != (indices.size,):
            raise ValueError("coefficients must have length N_active")

    def to_map(self, values, fill=0.0):
        """Scatter per-active-pixel values onto the pupil grid, keeping their units.

        Parameters
        ----------
        values : Array
            One value per active pixel, shape `(N_active,)`.
        fill : float, optional
            Value outside the active pixels. Default 0.

        Returns
        -------
        Array
            Map with the pupil grid's shape.
        """
        values = np.asarray(values)
        if values.shape != self.coefficients.shape:
            raise ValueError("values must have length N_active")
        flat = np.full(self.shape[0] * self.shape[1], fill, dtype=values.dtype)
        # Integer cast: the indices can come back as floats once inside a model.
        return flat.at[np.asarray(self.indices, dtype=int)].set(values).reshape(self.shape)

    @property
    def opd(self):
        """Full pupil OPD in metres; zero outside the fitted pixels."""
        return 1e-9 * self.to_map(self.coefficients)

    def apply(self, wavefront):
        """Add the OPD to a wavefront."""
        return wavefront.add_opd(self.opd)

    def __call__(self, wavefront):
        return self.apply(wavefront)


def nuisance_shapes(layer, terms=("tip", "tilt", "defocus")):
    """Orthonormal OPD shapes over a layer's active pixels, to marginalise.

    The shapes are orthonormalised (QR) together with piston, which is then dropped, so
    the columns are orthogonal to piston and to each other. Pass the result in
    `marginalise` (per exposure) or `marginalise_shared`.

    Parameters
    ----------
    layer : MaskedPixelOPD
        The layer whose active pixels define the shapes.
    terms : sequence, optional
        Names `"tip"` (x), `"tilt"` (y) and `"defocus"` (x² + y²), and/or arrays of
        length `N_active` for custom shapes. Default `("tip", "tilt", "defocus")`.

    Returns
    -------
    Array
        Shapes as columns, `(N_active, len(terms))`.

    Raises
    ------
    ValueError
        If a custom term does not have length `N_active`.
    """
    yy, xx = onp.indices(layer.shape, dtype=float)
    idx = onp.asarray(layer.indices).astype(int)
    x = (xx.ravel()[idx] - xx.ravel()[idx].mean()) / layer.shape[1]
    y = (yy.ravel()[idx] - yy.ravel()[idx].mean()) / layer.shape[0]
    named = {"tip": x, "tilt": y, "defocus": x**2 + y**2}
    columns = [onp.ones(idx.size)]
    for term in terms:
        column = named[term] if isinstance(term, str) else onp.asarray(term, dtype=float)
        if column.shape != (idx.size,):
            raise ValueError("custom terms must have length N_active")
        columns.append(column)
    return onp.linalg.qr(onp.column_stack(columns))[0][:, 1:]


def _linearise(model, key, q0, shape, noise, variance, variance_fn, bad):
    """Linearisation of one model's prediction in the parameter at `key`."""

    def predict(q):
        return np.asarray(model.set(key, q.reshape(shape)).model()).ravel()

    mu = predict(q0)
    weight = noise_weights(mu, noise=noise, variance=variance, variance_fn=variance_fn,
                           bad_mask=None if bad is None else onp.asarray(bad).ravel())

    @jax.jit
    def pushforward(v):
        return jax.jvp(predict, (q0,), (v,))[1]

    @jax.jit
    def pullback(t):
        return jax.vjp(predict, q0)[1](t)[0]

    return _Linearisation(model, weight, pushforward, pullback)


def wavefront_fisher(models, key, *, q0=None, noise="poisson", variance=1.0, variance_fn=None,
                     bad_masks=None, marginalise=None, marginalise_shared=None, rcond=1e-10):
    """Matrix-free Fisher matrix of the parameter at `key`, nuisances marginalised.

    The Fisher matrix is `sum J.T diag(1 / variance) J` over models and included pixels,
    where `J` is the Jacobian of each model's prediction with respect to the parameter at
    `key` (flattened). It is applied with one JVP and one VJP per model, so neither the
    matrix nor a dense Jacobian is formed. With `marginalise` or `marginalise_shared`, it
    is the Fisher matrix left once those nuisances are fitted freely (Schur complement);
    see `dluxtras.eigenmodes`. Other parameters are held fixed.

    Parameters
    ----------
    models : zodiax.Base or list
        A model or list/tuple of models (e.g. exposures) that share the parameter at
        `key`. Any output shape works, e.g. an `(n_exposures, ny, nx)` stack.
    key : str
        Path of the parameter, e.g. `"optics.fit_opd.coefficients"` (a
        `MaskedPixelOPD`) or `"optics.aberrations.coefficients"` (Zernikes).
    q0 : Array, optional
        Reference value of that parameter, used in every model. Default is its value in
        the first model.
    noise : {"poisson", "gaussian"}, optional
        `"poisson"` (default) uses the reference prediction as the variance (counts,
        strictly positive on included pixels); `"gaussian"` uses `variance`.
    variance : float or Array or list, optional
        Gaussian variance (not standard deviation), for every model or one per model in a
        list. Default 1.
    variance_fn : callable, optional
        `variance_fn(mu) -> variance` overrides both, e.g. `lambda mu: mu + read**2`.
    bad_masks : Array or list, optional
        Boolean mask of the model output (True excludes), for every model or one (or
        None) per model in a list.
    marginalise : list, optional
        Per-model nuisances: model keys and/or arrays of directions in the parameter's
        flattened space, `(n,)` or `(n, m)`, e.g. `nuisance_shapes(layer)`.
    marginalise_shared : list, optional
        Nuisances shared by all models, e.g. `["optics.psf_pixel_scale"]`.
    rcond : float, optional
        Relative cutoff of the nuisance pseudo-inverse (in unit-free form). Default 1e-10.

    Returns
    -------
    FisherOperator
        `F(v)` gives `F @ v`; `F.dense()` builds the matrix (small parameters only).

    Raises
    ------
    ValueError
        If `jax_enable_x64` is not enabled, the number of masks or variances does not
        match the number of models, or the noise model is invalid (see
        `eigenmodes.noise_weights`).
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before building models")
    models = list(models) if isinstance(models, (list, tuple)) else [models]
    if not models:
        raise ValueError("Supply a model or a list of models")
    q0 = np.asarray(models[0].get(key) if q0 is None else q0, dtype=float)
    shape = q0.shape
    q0 = q0.ravel()
    masks = _each(bad_masks, len(models), "bad_masks")
    variances = _each(variance, len(models), "variance")
    lins = [_linearise(m, key, q0, shape, noise, v, variance_fn, bad)
            for m, bad, v in zip(models, masks, variances)]
    return _assemble(lins, q0.size, marginalise, marginalise_shared, rcond)


@dataclass
class WavefrontModes:
    """Result of `wavefront_eigenmodes`: the Fisher operator and its leading eigenmodes.

    The fields shared with `spectral_eigenmodes.SpectralModes` (`reference`, `fisher`,
    `values`, `vectors`, `residuals`, `sigma`, `coefficients`) mean the same.

    Attributes
    ----------
    key : str
        Path of the parameter.
    reference : Array
        Reference value of the parameter (flattened), shape (n,).
    fisher : FisherOperator
        The (marginalised, if requested) Fisher matrix, matrix-free.
    values : Array
        Leading eigenvalues, descending, shape (k,).
    vectors : Array
        Unit-norm eigenvectors as columns, shape (n, k).
    residuals : Array
        Relative Lanczos residuals, shape (k,); check they are small.
    layer : MaskedPixelOPD or None
        The layer at `key`, if it is a `MaskedPixelOPD`; used by `maps`.
    """

    key: str
    reference: Any
    fisher: FisherOperator
    values: Any
    vectors: Any
    residuals: Any
    layer: Any = None

    @property
    def sigma(self):
        """1-sigma error of each mode's coefficient, in the parameter's units (nm for OPD).

        Conditional on the other modes being known (other parameters held fixed, or
        marginalised if requested).

        Returns
        -------
        Array
            `1 / sqrt(values)`, inf where the eigenvalue is not positive; shape (k,).
        """
        positive = self.values > 0
        return onp.where(positive, 1 / onp.sqrt(onp.where(positive, self.values, 1.0)),
                         onp.inf)

    def coefficients(self, q):
        """Project a parameter vector onto the modes.

        Parameters
        ----------
        q : Array
            Parameter value, any shape with `n` entries (e.g. an OPD per active pixel).

        Returns
        -------
        Array
            `vectors.T @ q`, shape (k,).
        """
        return self.vectors.T @ onp.asarray(q).ravel()

    def maps(self, fill=onp.nan):
        """Mode maps on the pupil grid (only for `MaskedPixelOPD` keys).

        Parameters
        ----------
        fill : float, optional
            Value outside the fitted pixels. Default NaN, for plotting.

        Returns
        -------
        Array
            `(k, ny, nx)` maps of the unit-norm eigenvectors.
        """
        if self.layer is None:
            raise ValueError("maps needs a MaskedPixelOPD at key")
        return mode_maps(self.layer, self.vectors, fill)


def wavefront_eigenmodes(models, key, k=6, *, q0=None, remove_piston=None, marginalise=None,
                         marginalise_shared=None, noise="poisson", variance=1.0,
                         variance_fn=None, bad_masks=None, rcond=1e-10, tol=1e-7, ncv=None,
                         seed=0):
    """Leading wavefront Fisher eigenmodes in one call.

    Runs `wavefront_fisher` and `eigenmodes.leading_modes`. The arguments match
    `spectral_eigenmodes.spectral_eigenmodes` where they overlap.

    Parameters
    ----------
    models : zodiax.Base or list
        Model(s) sharing the parameter at `key`; see `wavefront_fisher`.
    key : str
        Path of the parameter, e.g. `"optics.fit_opd.coefficients"`.
    k : int, optional
        Number of leading modes. Default 6.
    q0 : Array, optional
        Reference value; see `wavefront_fisher`.
    remove_piston : bool, optional
        Project out a constant (piston) direction. Default None: True when `key` is the
        `coefficients` of a `MaskedPixelOPD`, else False.
    marginalise, marginalise_shared : list, optional
        Nuisances to marginalise; see `wavefront_fisher`.
    noise, variance, variance_fn, bad_masks
        Noise model; see `wavefront_fisher`.
    rcond : float, optional
        See `wavefront_fisher`. Default 1e-10.
    tol, ncv, seed
        Lanczos settings; see `eigenmodes.leading_modes`.

    Returns
    -------
    WavefrontModes
    """
    models_list = list(models) if isinstance(models, (list, tuple)) else [models]
    fisher = wavefront_fisher(models, key, q0=q0, noise=noise, variance=variance,
                              variance_fn=variance_fn, bad_masks=bad_masks,
                              marginalise=marginalise, marginalise_shared=marginalise_shared,
                              rcond=rcond)
    reference = onp.asarray(models_list[0].get(key) if q0 is None else q0).ravel()
    layer = None
    if key.endswith(".coefficients"):
        candidate = models_list[0].get(key[: -len(".coefficients")])
        layer = candidate if isinstance(candidate, MaskedPixelOPD) else None
    if remove_piston is None:
        remove_piston = layer is not None
    project_out = onp.ones(reference.size) if remove_piston else None
    values, vectors, residuals = leading_modes(fisher, k, project_out=project_out, tol=tol,
                                               ncv=ncv, seed=seed)
    return WavefrontModes(key, reference, fisher, values, vectors, residuals, layer)


def mode_maps(layer, vectors, fill=onp.nan):
    """Scatter mode vectors onto the pupil grid.

    Parameters
    ----------
    layer : MaskedPixelOPD
        The layer whose active pixels the vectors live on.
    vectors : Array
        Modes as columns, `(N_active, k)`.
    fill : float, optional
        Value outside the active pixels. Default NaN, for plotting.

    Returns
    -------
    Array
        `(k, ny, nx)` maps.
    """
    return onp.stack([onp.asarray(layer.to_map(vectors[:, i], fill=fill))
                      for i in range(vectors.shape[1])])
