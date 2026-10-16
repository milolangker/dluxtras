"""Fisher-preconditioned fitting for dLux/Zodiax models.

A `FitProblem` holds what is fitted (model, parameters, data and likelihood). Its
`curvature` method computes the Fisher matrix in unit-free form, and its `fit` method
returns a resumable `FisherFit` that optimises in Fisher-whitened coordinates.

The three main ways to fit are:

```python
problem = FitProblem(model, params, data, gaussian_nll, std, trainable=mask)
fit = problem.fit().run(200)                        # 1. diagonal Fisher, Nesterov SGD
fit = problem.fit(optimiser="lbfgs").run(100)       # 2. L-BFGS (stops when converged)
fit = problem.fit(precondition="full").run(10)      # 3. natural gradient
sigma = problem.curvature(fit.params, precondition="full").std   # marginal 1-sigma
```

Enable `jax_enable_x64` before building models. See the "Fitting" guide in the
documentation for when to use each option.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as np
import jax.scipy.linalg as jsl
import numpy as onp
import optax
from jax.flatten_util import ravel_pytree

from .curvature import correlation, scaled_covariance, unscale

__all__ = ["FitProblem", "Curvature", "FisherFit", "ParameterSelection", "gaussian_nll"]


def gaussian_nll(prediction, data, std):
    """Independent Gaussian negative log likelihood, up to a constant.

    Pixels where `data` or `std` is NaN are excluded. Unlike `np.nansum`, the masking
    happens before any arithmetic, so gradients stay finite at excluded pixels.

    Parameters
    ----------
    prediction : Array
        Model prediction.
    data : Array
        Data, the same shape as `prediction`. NaN marks excluded pixels.
    std : Array
        Per-pixel standard deviation, broadcastable to `data`. NaN also excludes.

    Returns
    -------
    Array
        Scalar `0.5 * sum(((data - prediction) / std) ** 2)` over included pixels.
    """
    excluded = np.isnan(data) | np.isnan(std)
    residual = np.where(excluded, 0.0, np.where(excluded, 0.0, data) - prediction)
    weight = np.where(excluded, 0.0, 1 / np.where(excluded, 1.0, std) ** 2)
    return 0.5 * np.sum(weight * residual**2)


class ParameterSelection:
    """Flatten dictionary parameters and keep track of which entries are fitted.

    Used internally by `FitProblem`; exposed for its labels and helpers.

    Parameters
    ----------
    params : dict
        `{path: array}` of real floating parameters, e.g.
        `{"source.position": np.zeros(2)}`. All values must be finite.
    trainable : dict, optional
        `{path: bool or boolean array}`. `True` fits every entry of that key, `False`
        freezes it, and a boolean array (broadcastable to the parameter's shape)
        selects entries. Keys not given are fitted. Default fits everything.

    Attributes
    ----------
    initial_params : dict
        The parameters as given.
    initial_active : Array
        The fitted (active) entries, flattened in JAX's sorted-key order.
    indices : Array
        Positions of the active entries in the full flattened vector.
    active_labels : list of str
        One label per active entry, e.g. `"source.position[0]"`.
    mask : dict
        The boolean mask per key, at full shape.
    """

    def __init__(self, params, trainable=None):
        if not isinstance(params, dict) or not params:
            raise ValueError("params must be a nonempty dictionary of real arrays")
        trainable = {} if trainable is None else trainable
        unknown = set(trainable) - set(params)
        if unknown:
            raise ValueError(f"Unknown mask keys: {sorted(unknown)}")

        self.initial_params = {}
        masks = {}
        for key, value in params.items():
            value = np.asarray(value)
            if not np.issubdtype(value.dtype, np.floating):
                raise TypeError(f"{key}: parameters must be real floating arrays")
            if not onp.isfinite(onp.asarray(value)).all():
                raise ValueError(f"{key}: initial parameters must be finite")
            self.initial_params[key] = value
            raw = onp.asarray(trainable.get(key, True))
            if raw.dtype != onp.dtype(bool):
                raise TypeError(f"{key}: trainable mask must be boolean")
            masks[key] = np.asarray(onp.broadcast_to(raw, value.shape))

        self.initial_flat, self.unflatten = ravel_pytree(self.initial_params)
        mask_flat, _ = ravel_pytree(masks)
        self.indices = np.asarray(onp.flatnonzero(onp.asarray(mask_flat)))
        if not self.indices.size:
            raise ValueError("Select at least one trainable parameter entry")
        self.mask = masks
        self.initial_active = self.initial_flat[self.indices]
        # Dict leaves are flattened in sorted key order by JAX.
        labels = [
            key if value.ndim == 0 else f"{key}[{', '.join(map(str, index))}]"
            for key in sorted(self.initial_params)
            for value in [self.initial_params[key]]
            for index in onp.ndindex(value.shape)
        ]
        self.active_labels = [labels[int(i)] for i in self.indices]

    def active_of(self, params):
        """Active (fitted) entries of a full parameter dictionary.

        Parameters
        ----------
        params : dict
            Dictionary with the same keys and shapes as the initial parameters.

        Returns
        -------
        Array
            The active entries, flattened, ordered as `active_labels`.
        """
        if set(params) != set(self.initial_params):
            raise ValueError(f"params must have keys {sorted(self.initial_params)}")
        flat, _ = ravel_pytree({key: np.asarray(params[key]) for key in self.initial_params})
        if flat.shape != self.initial_flat.shape:
            raise ValueError("params shapes differ from the problem's")
        return flat[self.indices]

    def expand(self, active):
        """Full parameter dictionary from active entries; frozen entries unchanged.

        Parameters
        ----------
        active : Array
            Values for the active entries, ordered as `active_labels`.

        Returns
        -------
        dict
            `{path: array}` with the initial values for frozen entries.
        """
        return self.unflatten(self.initial_flat.at[self.indices].set(active))

    def expand_zeros(self, active):
        """Like `expand`, but frozen entries are set to exactly zero.

        Used for per-entry quantities such as uncertainties.

        Parameters
        ----------
        active : Array
            Values for the active entries, ordered as `active_labels`.

        Returns
        -------
        dict
            `{path: array}` with zeros for frozen entries.
        """
        return self.unflatten(np.zeros_like(self.initial_flat).at[self.indices].set(active))


def _batched(fn, n, batch_size, dtype):
    """Stack fn(e_i) over unit vectors e_i, batch_size at a time."""
    fn = jax.jit(jax.vmap(fn))
    basis = np.eye(n, dtype=dtype)
    return np.concatenate([fn(basis[i : i + batch_size]) for i in range(0, n, batch_size)])


def _largest_eigenvalue(matvec, n, iterations=20):
    """Power-iteration estimate (from below) of a PSD operator's largest eigenvalue."""
    v = jax.random.normal(jax.random.PRNGKey(0), (n,))
    v = v / np.linalg.norm(v)
    value = 0.0
    for _ in range(iterations):
        w = matvec(v)
        value = float(np.linalg.norm(w))
        v = w / value
    return value


def _apply(transform, u):
    return transform * u if transform.ndim == 1 else transform @ u


class FitProblem:
    """What is fitted: model, parameters, data and likelihood.

    Parameters
    ----------
    model : zodiax.Base
        The model, e.g. a `dl.Telescope`. Parameters are set with `model.set(params)`.
    params : dict
        Starting parameters, `{path: array}`.
    data : Array
        The data. NaN marks excluded pixels (the likelihood must handle them).
    nll : callable
        `nll(prediction, data, *nll_args, **nll_kwargs) -> scalar`, a negative log
        likelihood: summed over pixels, not averaged, so that the Fisher matrix and
        uncertainties come out in the right units. It must give finite gradients at
        NaN pixels; mask with `np.where` before arithmetic, as `gaussian_nll` does.
    *nll_args
        Extra positional arguments for `nll`, e.g. the per-pixel `std`.
    trainable : dict, optional
        Which entries to fit; see `ParameterSelection`. Default fits everything.
    predict : callable, optional
        `predict(params, model) -> prediction`. Default `model.set(params).model()`.
    nll_kwargs : dict, optional
        Extra keyword arguments for `nll`.

    Raises
    ------
    ValueError
        If `jax_enable_x64` is not enabled.
    """

    def __init__(self, model, params, data, nll, *nll_args, trainable=None, predict=None,
                 nll_kwargs=None):
        if not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 before constructing your models")
        self.model = model
        self.data = np.asarray(data)
        self.selection = ParameterSelection(params, trainable)
        self.predict_fn = (lambda p, m: m.set(p).model()) if predict is None else predict
        kwargs = {} if nll_kwargs is None else dict(nll_kwargs)
        self.nll = lambda prediction, data: nll(prediction, data, *nll_args, **kwargs)

    @property
    def active_labels(self):
        """Labels of the fitted entries, e.g. `"source.position[0]"`."""
        return self.selection.active_labels

    def predict(self, active):
        """Model prediction for active entries `active` (frozen entries unchanged)."""
        return np.asarray(self.predict_fn(self.selection.expand(active), self.model))

    def loss(self, active, data=None):
        """The negative log likelihood at active entries `active`.

        Parameters
        ----------
        active : Array
            Values of the active entries, ordered as `active_labels`.
        data : Array, optional
            Data to evaluate against. Default is the problem's data.

        Returns
        -------
        Array
            Scalar loss.
        """
        return self.nll(self.predict(active), self.data if data is None else data)

    def curvature(self, params=None, *, precondition="diagonal", kind="expected", damping=0.0,
                  curvature_data=None, batch_size=1):
        """The Fisher matrix (or loss Hessian) at `params`, in unit-free form.

        With `kind="expected"` this is the Fisher matrix `F = J.T H J`, where `J` is
        the Jacobian of the prediction and `H` the Hessian of `nll` with respect to
        the prediction, with the data replaced by the prediction (NaN pixels kept).
        That is exactly the Fisher matrix for Gaussian noise of fixed (co)variance,
        correlated or not, and for Poisson noise, and it is positive semi-definite
        everywhere.

        Parameters
        ----------
        params : dict, optional
            Where to evaluate it. Default is the problem's initial parameters. Frozen
            entries always take the problem's values.
        precondition : {"diagonal", "full"}, optional
            `"diagonal"` (default) computes only the diagonal of `F`, plus its
            largest normalised eigenvalue by power iteration; that is enough for
            whitening but not for covariances. `"full"` computes the whole matrix.
        kind : {"expected", "observed"}, optional
            `"expected"` (default) is the Fisher matrix described above. `"observed"`
            is the Hessian of the loss against the actual data (or `curvature_data`),
            which includes residual terms and can be indefinite away from the best
            fit. `"observed"` always computes the full matrix.
        damping : float, optional
            Added to the normalised matrix's diagonal before whitening, so 0.01 adds
            1% of each diagonal entry. Only for `precondition="full"`, to whiten near-
            singular problems. Covariances stay undamped. Default 0.
        curvature_data : Array, optional
            Data to use instead of the problem's data. Only for `kind="observed"`.
        batch_size : int, optional
            How many derivative columns to compute at once. Affects memory only.
            Default 1.

        Returns
        -------
        Curvature

        Raises
        ------
        ValueError
            If a parameter has no positive curvature (freeze it), if the curvature is
            not finite, or if `precondition="full"` and the normalised matrix is
            singular or indefinite (freeze degenerate parameters or set `damping`).
        """
        if precondition not in ("diagonal", "full"):
            raise ValueError("precondition must be 'diagonal' or 'full'")
        if kind not in ("expected", "observed"):
            raise ValueError("kind must be 'expected' or 'observed'")
        if not (onp.isfinite(damping) and damping >= 0):
            raise ValueError("damping must be finite and nonnegative")
        if damping and precondition == "diagonal":
            raise ValueError("damping only applies to precondition='full'")
        if curvature_data is not None and kind == "expected":
            raise ValueError("curvature_data only applies to kind='observed'")
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        x = self.selection.initial_active if params is None else self.selection.active_of(params)
        n = x.size
        if kind == "expected":
            mu, jvp = jax.linearize(self.predict, x)
            reference = np.where(np.isnan(self.data), np.nan, mu)
            nll_grad = jax.grad(lambda m: self.nll(m, reference))

            def hvp(t):  # Hessian of nll in prediction space, times t
                return jax.jvp(nll_grad, (mu,), (t,))[1]

            if precondition == "diagonal":
                def column_curvature(v):
                    t = jvp(v)
                    return np.sum(t * hvp(t))

                diagonal = _batched(column_curvature, n, batch_size, x.dtype)
                matrix = None
            else:
                pullback = jax.vjp(self.predict, x)[1]
                matrix = _batched(lambda v: pullback(hvp(jvp(v)))[0], n, batch_size, x.dtype)
        else:
            data = self.data if curvature_data is None else np.asarray(curvature_data)
            loss_grad = jax.grad(lambda a: self.loss(a, data))
            matrix = _batched(lambda v: jax.jvp(loss_grad, (x,), (v,))[1], n, batch_size,
                              x.dtype)
        if matrix is not None:
            matrix = (matrix + matrix.T) / 2
            diagonal = np.diag(matrix)
        if not bool(np.all(np.isfinite(diagonal))):
            raise ValueError("Nonfinite curvature: check that nll has finite gradients at "
                             "NaN pixels (see gaussian_nll)")
        bad = [label for label, d in zip(self.active_labels, onp.asarray(diagonal)) if not d > 0]
        if bad:
            raise ValueError(f"No positive curvature (freeze these parameters): {bad}")
        scales = 1 / np.sqrt(diagonal)

        normalised = eigenvalues = None
        if matrix is not None:
            normalised = scales[:, None] * matrix * scales[None, :]
            eigenvalues = np.linalg.eigvalsh(normalised)
            lambda_max = float(eigenvalues[-1])
        else:
            pullback = jax.vjp(self.predict, x)[1]
            fvp = jax.jit(lambda v: scales * pullback(hvp(jvp(scales * v)))[0])
            lambda_max = _largest_eigenvalue(fvp, n)

        if precondition == "diagonal":
            transform = scales / np.sqrt(lambda_max)
        else:
            damped = normalised + damping * np.eye(n, dtype=x.dtype)
            values = np.linalg.eigvalsh(damped)
            if not float(values[0]) > 1e-12 * float(values[-1]):
                raise ValueError(
                    f"Normalised curvature is singular or indefinite (eigenvalues "
                    f"{float(values[0]):.3g} to {float(values[-1]):.3g}). Freeze degenerate "
                    "parameters, use kind='expected', or set damping > 0."
                )
            inverse_factor = jsl.solve_triangular(
                np.linalg.cholesky(damped), np.eye(n, dtype=x.dtype), lower=True
            )
            transform = scales[:, None] * inverse_factor.T

        return Curvature(self.selection, x, kind, precondition, damping, curvature_data,
                         batch_size, scales, normalised, eigenvalues, lambda_max, transform)

    def fit(self, curvature=None, optimiser=None, tol=None, **curvature_options):
        """Start a resumable `FisherFit` where the curvature was evaluated.

        Parameters
        ----------
        curvature : Curvature, optional
            The curvature to whiten with. Default computes one with
            `self.curvature(**curvature_options)`.
        optimiser : None, "lbfgs" or optax.GradientTransformation, optional
            `None` (default) picks SGD with lr 1: with Nesterov momentum 0.9 for a
            diagonal curvature, plain (natural gradient) for a full one. `"lbfgs"` is
            `optax.lbfgs()`. Any optax transform also works; it receives gradients
            in whitened coordinates. Avoid `optax.adam`, which undoes the whitening.
        tol : float, optional
            Stop once steps move every parameter by less than `tol` conditional sigma;
            see `FisherFit`. Default 0.01 for `"lbfgs"` and 0 (no early stop) otherwise.
        **curvature_options
            Passed to `curvature` when no `curvature` is given, e.g.
            `precondition="full"`.

        Returns
        -------
        FisherFit

        Raises
        ------
        ValueError
            If both `curvature` and `curvature_options` are given.
        """
        if curvature is not None and curvature_options:
            raise ValueError("Pass either a Curvature or curvature options, not both")
        if curvature is None:
            curvature = self.curvature(**curvature_options)
        return FisherFit(self, curvature, optimiser, tol)


@dataclass
class Curvature:
    """The Fisher matrix (or loss Hessian) at one point, held in unit-free form.

    Created by `FitProblem.curvature`. The matrix is stored as
    `F = normalised / outer(scales, scales)`, where `normalised` has a unit diagonal.
    Matrix-valued properties need `precondition="full"` (or `kind="observed"`).

    Attributes
    ----------
    selection : ParameterSelection
        The problem's parameter selection.
    active : Array
        The active entries where the curvature was evaluated.
    kind, precondition : str
        As passed to `FitProblem.curvature`.
    damping : float
        As passed to `FitProblem.curvature`.
    curvature_data : Array or None
        As passed to `FitProblem.curvature`.
    batch_size : int
        As passed to `FitProblem.curvature`.
    scales : Array
        `1 / sqrt(diag(F))`: the conditional 1-sigma of each active entry (all other
        entries held fixed), in the parameters' own units.
    normalised : Array or None
        `S F S` with `S = diag(scales)`, unit diagonal. `None` for a diagonal-only
        curvature.
    eigenvalues : Array or None
        Eigenvalues of `normalised`, ascending. `None` for a diagonal-only curvature.
    lambda_max : float
        Largest eigenvalue of `normalised` (power iteration if diagonal only).
    transform : Array
        `T` in `x = x0 + T u`, the whitening: a vector for diagonal, a matrix for full.
    """

    selection: ParameterSelection
    active: Any
    kind: str
    precondition: str
    damping: float
    curvature_data: Any
    batch_size: int
    scales: Any
    normalised: Any
    eigenvalues: Any
    lambda_max: float
    transform: Any

    @property
    def labels(self):
        """Labels of the active entries; the order of all matrices and vectors."""
        return self.selection.active_labels

    @property
    def params(self):
        """Full parameter dictionary at which this curvature was evaluated."""
        return self.selection.expand(self.active)

    @property
    def conditional_std(self):
        """Conditional 1-sigma, `1 / sqrt(F_ii)`, in the parameters' dict shapes.

        This is the uncertainty with all other entries held fixed, so it is
        optimistic when parameters are correlated. Frozen entries are 0.
        """
        return self.selection.expand_zeros(self.scales)

    def _full(self):
        if self.normalised is None:
            raise ValueError("Needs the full matrix: compute with precondition='full'")
        return self.normalised

    @property
    def matrix(self):
        """The matrix `F` in the parameters' own units, ordered by `labels`.

        In float32 this can over/underflow when units span many decades; prefer
        `normalised` and `scales` for any further linear algebra.
        """
        return unscale(self._full(), self.scales)

    @property
    def covariance(self):
        """Covariance `inv(F)` in the parameters' own units, ordered by `labels`.

        Computed by Cholesky of `normalised`, then rescaled; never damped.

        Raises
        ------
        ValueError
            If the curvature is not positive definite.
        """
        covariance = scaled_covariance(self._full(), self.scales)
        if not bool(np.all(np.isfinite(covariance))):
            raise ValueError("Curvature is not positive definite; no covariance exists")
        return covariance

    @property
    def std(self):
        """Marginal 1-sigma, `sqrt(diag(covariance))`, in the parameters' dict shapes.

        Frozen entries are 0; they are treated as known, not marginalised over.
        """
        return self.selection.expand_zeros(np.sqrt(np.diag(self.covariance)))

    @property
    def correlation(self):
        """Correlation matrix of the covariance, ordered by `labels`."""
        return correlation(self._full())

    @property
    def condition_number(self):
        """Condition number of `normalised`: the degeneracy left once units are removed.

        Roughly, digits lost when inverting. Values near 1 mean uncorrelated
        parameters; large values mean near-degenerate combinations.
        """
        self._full()
        return float(self.eigenvalues[-1] / self.eigenvalues[0])


def _make_optimiser(optimiser, precondition):
    if optimiser is None:
        if precondition == "diagonal":
            return optax.sgd(1.0, momentum=0.9, nesterov=True)
        return optax.sgd(1.0)
    if optimiser == "lbfgs":
        return optax.lbfgs()
    if not hasattr(optimiser, "update"):
        raise ValueError("optimiser must be None, 'lbfgs', or an optax transform "
                         "(e.g. optax.sgd(1.0) for plain SGD)")
    return optimiser


class FisherFit:
    """A resumable fit in Fisher-whitened coordinates.

    Usually created by `FitProblem.fit`. The fit optimises `u`, where the active
    parameters are `x = x0 + T u` with `T = curvature.transform`, so that the curvature
    in `u` at `x0` is (close to) the identity. `run` and `refresh` return `self`, so
    calls chain: `fit.run(200).refresh().run(200)`.

    Parameters
    ----------
    problem : FitProblem
        The problem to fit.
    curvature : Curvature
        The curvature to whiten with; the fit starts where it was evaluated.
    optimiser : None, "lbfgs" or optax.GradientTransformation, optional
        See `FitProblem.fit`.
    tol : float, optional
        `run` stops once two consecutive steps each move every parameter by less than
        `tol` conditional sigma (`curvature.scales`), and sets `converged`. Because it
        is measured in sigma, the test does not depend on the likelihood's
        normalisation, provided `nll` is a proper (summed) negative log likelihood.
        Default 0.01 for `"lbfgs"` and 0 (no early stop) otherwise. Small steps
        reliably mean convergence for L-BFGS and natural gradient, whose steps approach
        the whole remaining distance. Diagonal SGD on correlated parameters can take
        small steps while still far away, so use a smaller `tol` there.

    Attributes
    ----------
    losses : list of float
        Loss before each step.
    history : list of Array
        Active entries before each step, so `history[i]` produced `losses[i]`.
    moves : list of float
        Size of each step: the largest change of any parameter, in conditional sigma.
    steps : int
        Number of steps taken.
    converged : bool
        Whether the `tol` test has been met (reset by `refresh`).
    curvature : Curvature
        The curvature currently used for whitening.
    state
        The optimiser state.
    """

    def __init__(self, problem, curvature, optimiser=None, tol=None):
        self.problem = problem
        if tol is None:
            tol = 0.01 if isinstance(optimiser, str) and optimiser == "lbfgs" else 0.0
        if not (onp.isfinite(tol) and tol >= 0):
            raise ValueError("tol must be finite and nonnegative")
        self.tol = float(tol)
        self.optimiser = _make_optimiser(optimiser, curvature.precondition)
        self.losses, self.history, self.moves, self.steps = [], [], [], 0
        loss = problem.loss
        optimiser = self.optimiser
        # Line-search optimisers (L-BFGS) store the value and gradient at the accepted
        # point; reusing them avoids recomputing both at the start of the next step.
        probe = optimiser.init(np.zeros_like(curvature.active))
        reuse = all(optax.tree_utils.tree_get(probe, name) is not None
                    for name in ("value", "grad"))

        @jax.jit
        def step(u, state, center, transform, scales):
            def value(v):
                return loss(center + _apply(transform, v))

            if reuse:
                loss_value, grad = optax.value_and_grad_from_state(value)(u, state=state)
            else:
                loss_value, grad = jax.value_and_grad(value)(u)
            updates, state = optimiser.update(grad, state, u, value=loss_value, grad=grad,
                                              value_fn=value)
            new_u = optax.apply_updates(u, updates)
            finite_loss = np.isfinite(loss_value) & np.all(np.isfinite(grad))
            finite_step = np.all(np.isfinite(new_u))
            move = np.max(np.abs(_apply(transform, new_u - u)) / scales)
            return loss_value, new_u, state, finite_loss, finite_step, move

        self._step = step
        self._restart(curvature)

    def _restart(self, curvature):
        self.curvature = curvature
        self._u = np.zeros_like(curvature.active)
        self.state = self.optimiser.init(self._u)
        self.converged, self._small_steps = False, 0

    @property
    def active(self):
        """Current values of the active entries, ordered as `active_labels`."""
        return self.curvature.active + _apply(self.curvature.transform, self._u)

    @property
    def active_labels(self):
        """Labels of the active entries."""
        return self.problem.active_labels

    @property
    def params(self):
        """Current full parameter dictionary, frozen entries included."""
        return self.problem.selection.expand(self.active)

    @property
    def fitted_model(self):
        """The model with the current parameters set."""
        return self.problem.model.set(self.params)

    @property
    def active_history(self):
        """Active entries before each step, shape `(n_steps, n_active)`.

        Columns are ordered as `active_labels`, so
        `plt.plot(fit.active_history, label=fit.active_labels)` plots one line each.
        """
        if not self.history:
            return np.zeros((0, self.active.size), dtype=self.active.dtype)
        return np.stack(self.history)

    @property
    def param_history(self):
        """Full parameter dictionary before each step, `{key: (n_steps, *shape)}`."""
        return jax.vmap(self.problem.selection.expand)(self.active_history)

    def refresh(self):
        """Recompute the curvature at the current parameters, with the same settings.

        The fit restarts from there with a fresh optimiser state; history, losses and
        step count are kept.

        Returns
        -------
        FisherFit
            `self`.
        """
        c = self.curvature
        self._restart(self.problem.curvature(
            self.params, precondition=c.precondition, kind=c.kind, damping=c.damping,
            curvature_data=c.curvature_data, batch_size=c.batch_size))
        return self

    def step(self):
        """Take one step, even if already converged.

        Returns
        -------
        float
            The loss before the step.

        Raises
        ------
        FloatingPointError
            If the loss, gradient or proposed parameters are not finite. No update is
            applied, so the fit can continue from the last good point.
        """
        c = self.curvature
        loss, u, state, finite_loss, finite_step, move = self._step(
            self._u, self.state, c.active, c.transform, c.scales)
        if not bool(finite_loss):
            raise FloatingPointError("Nonfinite loss/gradient; no update applied")
        if not bool(finite_step):
            raise FloatingPointError("Nonfinite proposed parameters; no update applied")
        self.history.append(self.active)
        self._u, self.state = u, state
        self.steps += 1
        self.losses.append(float(loss))
        self.moves.append(float(move))
        self._small_steps = self._small_steps + 1 if self.moves[-1] < self.tol else 0
        self.converged = self._small_steps >= 2
        return self.losses[-1]

    def run(self, nsteps=100, *, progress=True):
        """Take up to `nsteps` steps, stopping early once converged (see `tol`).

        Parameters
        ----------
        nsteps : int, optional
            Maximum number of steps. Default 100.
        progress : bool, optional
            Show a tqdm progress bar, if tqdm is installed. Default True.

        Returns
        -------
        FisherFit
            `self`.
        """
        if not isinstance(nsteps, int) or nsteps < 0:
            raise ValueError("nsteps must be a nonnegative integer")
        iterator = range(nsteps)
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator)
            except ImportError:
                pass
        for _ in iterator:
            if self.converged:
                break
            loss = self.step()
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{loss:.6g}", step=self.steps,
                                     move=f"{self.moves[-1]:.2g} sigma")
        if hasattr(iterator, "close"):
            iterator.close()
        return self
