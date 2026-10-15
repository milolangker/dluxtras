"""Fisher-preconditioned fitting for dLux/Zodiax models.

    problem = FitProblem(model, params, data, gaussian_nll, std, trainable=mask)
    fit = problem.fit()                      # diagonal Fisher, SGD with lr 1
    fit.run(200).refresh().run(200)          # refresh: new curvature where the fit is now
    fit = problem.fit(precondition="full", optimiser="lbfgs")   # fastest, most robust
    problem.curvature(fit.params, precondition="full").std   # marginal 1-sigma, dict shapes

Contract: nll(prediction, data, *nll_args, **nll_kwargs) -> scalar negative log
likelihood, where prediction = predict(params, model), by default
model.set(params).model(). params is a dict {path: array}; trainable selects
entries (True/False or a boolean array per key; unspecified keys are fitted) and
frozen entries keep their initial values exactly. nll must give finite
gradients at NaN (excluded) data: mask with np.where before arithmetic, as
gaussian_nll does; np.nansum alone does not. Enable jax_enable_x64 first.

Curvature (kind="expected", the default) is the Fisher matrix
F = J.T H J, with J = d prediction / d params and H the Hessian of nll with
respect to the prediction, evaluated with the data replaced by the prediction
(NaN pixels kept). This is exactly the Fisher matrix for Gaussian noise of fixed
(co)variance, correlated or not, and for Poisson noise, and it is positive
semi-definite everywhere. kind="observed" is the Hessian of the loss against
the actual data (or curvature_data): it includes residual terms and can be
indefinite away from the best fit.

Fits run in whitened coordinates u, with params x = x0 + T u, chosen so that
the curvature in u at x0 is (close to) the identity. Linear algebra is done on
the unit-free normalised Fisher F_n = S F S, S = diag(1/sqrt(diag(F))).

precondition="diagonal" (default): T = S / sqrt(lambda_max(F_n)). Only the
    diagonal of F is computed, and lambda_max by power iteration (kind="observed"
    always computes the full matrix, at the same cost). SGD is
    gradient descent preconditioned by 1/diag(F); lr = 1 is a Newton step along
    the stiffest direction and lr < 2 is stable at x0. Correlated directions
    converge at about lambda_min/lambda_max per step.
precondition="full": T = S L^-T with L L^T = F_n + damping I. SGD is natural-
    gradient descent with the Fisher fixed at x0: lr = 1 is a Newton (Fisher
    scoring) step, lr < 2 is stable at x0. Requires F_n to be non-singular, or
    damping > 0 (in unit-free terms: 0.01 adds 1% of each diagonal).

optimiser:
    1.0 (default): plain SGD with that learning rate; strictly first-order, one
        gradient per step. With the diagonal, lr = 1 steps exactly to the minimum
        along the stiffest direction; correlated parameters converge slowly.
    "lbfgs": optax.lbfgs, a line search with no learning rate, which also learns
        how the curvature changes as the fit moves. About 2 gradients per step,
        but in benchmarks/fitting_benchmark.md a whole fit took 3-17x fewer
        gradients than the default unless already near the minimum of an
        uncorrelated problem (then equal). With precondition="full" it was the
        fastest and most robust method from both near and far starts.
    "nesterov": SGD with lr 1 and Nesterov momentum 0.9. Still one gradient
        per step and no curvature model, but 5-9x fewer steps than plain SGD
        when parameters are correlated (similar when they are not).
    Any optax GradientTransformation also works; it sees u-space gradients.
    Avoid optax.adam: its per-coordinate rescaling undoes the whitening.

The curvature is computed once per fit; refresh() recomputes it at the current
parameters and restarts the optimiser state there.
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
    """Independent Gaussian negative log likelihood (up to a constant).

    Pixels where data or std is NaN are excluded, with finite gradients.
    """
    excluded = np.isnan(data) | np.isnan(std)
    residual = np.where(excluded, 0.0, np.where(excluded, 0.0, data) - prediction)
    weight = np.where(excluded, 0.0, 1 / np.where(excluded, 1.0, std) ** 2)
    return 0.5 * np.sum(weight * residual**2)


class ParameterSelection:
    """Flatten dictionary parameters and retain only trainable entries."""

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
        """Trainable entries of a full parameter dictionary."""
        if set(params) != set(self.initial_params):
            raise ValueError(f"params must have keys {sorted(self.initial_params)}")
        flat, _ = ravel_pytree({key: np.asarray(params[key]) for key in self.initial_params})
        if flat.shape != self.initial_flat.shape:
            raise ValueError("params shapes differ from the problem's")
        return flat[self.indices]

    def expand(self, active):
        """Full parameter dictionary with frozen entries unchanged."""
        return self.unflatten(self.initial_flat.at[self.indices].set(active))

    def expand_zeros(self, active):
        """Full-shaped values; frozen entries are exactly zero."""
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
    """What is fitted: model, parameters, data and likelihood."""

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
        return self.selection.active_labels

    def predict(self, active):
        return np.asarray(self.predict_fn(self.selection.expand(active), self.model))

    def loss(self, active, data=None):
        return self.nll(self.predict(active), self.data if data is None else data)

    def curvature(self, params=None, *, precondition="diagonal", kind="expected", damping=0.0,
                  curvature_data=None, batch_size=1):
        """Curvature at params (default: the problem's initial params).

        Frozen entries always take the problem's values. batch_size sets how many
        derivative columns are computed at once (memory only).
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

    def fit(self, curvature=None, optimiser=1.0, **curvature_options):
        """A resumable FisherFit starting where curvature was evaluated.

        Pass a Curvature, or options for self.curvature() (not both).
        optimiser: SGD learning rate (default 1.0), "lbfgs", "nesterov", or an optax
        transform.
        """
        if curvature is not None and curvature_options:
            raise ValueError("Pass either a Curvature or curvature options, not both")
        if curvature is None:
            curvature = self.curvature(**curvature_options)
        return FisherFit(self, curvature, optimiser)


@dataclass
class Curvature:
    """Fisher (or loss Hessian) at one point, held in unit-free form.

    scales: 1/sqrt(diag F), the conditional 1-sigma of each active entry.
    normalised: S F S with unit diagonal (precondition='full' or kind='observed').
    eigenvalues: of normalised, ascending (None when only the diagonal exists).
    lambda_max: largest eigenvalue of normalised (power iteration if diagonal only).
    transform: T in x = x0 + T u (vector for diagonal, matrix for full).
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
        return self.selection.active_labels

    @property
    def params(self):
        """Full parameter dictionary at which this curvature was evaluated."""
        return self.selection.expand(self.active)

    @property
    def conditional_std(self):
        """1/sqrt(F_ii) in dict shapes: 1-sigma with all other entries held fixed."""
        return self.selection.expand_zeros(self.scales)

    def _full(self):
        if self.normalised is None:
            raise ValueError("Needs the full matrix: compute with precondition='full'")
        return self.normalised

    @property
    def matrix(self):
        """F in original units, ordered by labels."""
        return unscale(self._full(), self.scales)

    @property
    def covariance(self):
        """inv(F) in original units (undamped), ordered by labels."""
        covariance = scaled_covariance(self._full(), self.scales)
        if not bool(np.all(np.isfinite(covariance))):
            raise ValueError("Curvature is not positive definite; no covariance exists")
        return covariance

    @property
    def std(self):
        """Marginal 1-sigma, sqrt(diag(covariance)), in dict shapes (frozen = 0)."""
        return self.selection.expand_zeros(np.sqrt(np.diag(self.covariance)))

    @property
    def correlation(self):
        """Correlation matrix of the covariance, ordered by labels."""
        return correlation(self._full())

    @property
    def condition_number(self):
        """cond(normalised): degeneracy left after removing units."""
        self._full()
        return float(self.eigenvalues[-1] / self.eigenvalues[0])


def _make_optimiser(optimiser):
    if isinstance(optimiser, str):
        named = {"lbfgs": optax.lbfgs, "nesterov": lambda: optax.sgd(1.0, 0.9, nesterov=True)}
        if optimiser not in named:
            raise ValueError("optimiser must be a learning rate, 'lbfgs', 'nesterov', or an "
                             "optax transform")
        return named[optimiser]()
    if isinstance(optimiser, (int, float)):
        if not (onp.isfinite(optimiser) and optimiser > 0):
            raise ValueError("learning rate must be finite and positive")
        return optax.sgd(float(optimiser))
    return optimiser


class FisherFit:
    """A resumable fit in whitened coordinates. run() mutates and returns self.

    Losses and history are recorded BEFORE each update, so history[i]
    produced losses[i]. active_history is (n_steps, n_active), columns named by
    active_labels; param_history is {key: (n_steps, *shape)}.
    """

    def __init__(self, problem, curvature, optimiser=1.0):
        self.problem = problem
        self.optimiser = _make_optimiser(optimiser)
        self.losses, self.history, self.steps = [], [], 0
        loss = problem.loss
        optimiser = self.optimiser

        @jax.jit
        def step(u, state, center, transform):
            def value(v):
                return loss(center + _apply(transform, v))

            loss_value, grad = jax.value_and_grad(value)(u)
            updates, state = optimiser.update(grad, state, u, value=loss_value, grad=grad,
                                              value_fn=value)
            return loss_value, grad, optax.apply_updates(u, updates), state

        self._step = step
        self._restart(curvature)

    def _restart(self, curvature):
        self.curvature = curvature
        self._u = np.zeros_like(curvature.active)
        self.state = self.optimiser.init(self._u)

    @property
    def active(self):
        return self.curvature.active + _apply(self.curvature.transform, self._u)

    @property
    def active_labels(self):
        return self.problem.active_labels

    @property
    def params(self):
        return self.problem.selection.expand(self.active)

    @property
    def fitted_model(self):
        return self.problem.model.set(self.params)

    @property
    def active_history(self):
        """Fitted entries before each step, shape (n_steps, n_active)."""
        if not self.history:
            return np.zeros((0, self.active.size), dtype=self.active.dtype)
        return np.stack(self.history)

    @property
    def param_history(self):
        """Full parameter dictionary before each step: {key: (n_steps, *shape)}."""
        return jax.vmap(self.problem.selection.expand)(self.active_history)

    def refresh(self):
        """Recompute the curvature (same settings) at the current parameters."""
        c = self.curvature
        self._restart(self.problem.curvature(
            self.params, precondition=c.precondition, kind=c.kind, damping=c.damping,
            curvature_data=c.curvature_data, batch_size=c.batch_size))
        return self

    def step(self):
        c = self.curvature
        loss, grad, u, state = self._step(self._u, self.state, c.active, c.transform)
        if not bool(np.isfinite(loss)) or not bool(np.all(np.isfinite(grad))):
            raise FloatingPointError("Nonfinite loss/gradient; no update applied")
        if not bool(np.all(np.isfinite(u))):
            raise FloatingPointError("Nonfinite proposed parameters; no update applied")
        self.history.append(self.active)
        self._u, self.state = u, state
        self.steps += 1
        self.losses.append(float(loss))
        return float(loss)

    def run(self, nsteps=100, *, progress=True):
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
            loss = self.step()
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{loss:.6g}", step=self.steps)
        return self
