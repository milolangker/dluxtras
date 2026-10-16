# Fitting

`dluxtras.fitting` fits dLux (or any Zodiax) models by optimising in coordinates whitened by
the Fisher matrix. This removes the problem of parameters with wildly different units, such as
Zernike coefficients in metres next to a flux of 1e6. It also gives you uncertainties from the
same machinery.

Three objects do the work:

- **`FitProblem`**: what is fitted, i.e. the model, the parameters, the data and the likelihood.
- **`Curvature`**: the Fisher matrix at one point, held in unit-free form. It provides the
  whitening and the uncertainties.
- **`FisherFit`**: a resumable optimisation that records its history.

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)  # required

from dluxtras import FitProblem, gaussian_nll

params = {"source.position": np.zeros(2), "aberrations.coefficients": np.zeros(8)}
problem = FitProblem(telescope, params, data, gaussian_nll, std)

fit = problem.fit().run(200)                     # default
fit = problem.fit(optimiser="lbfgs").run(100)    # usually fastest; stops when converged
fit = problem.fit(precondition="full").run(10)   # natural gradient, near the minimum

sigma = problem.curvature(fit.params, precondition="full").std   # marginal 1-sigma
best = fit.fitted_model
```

## Choosing a method

| | Call | Per step | Use when |
|---|---|---|---|
| 1 | `problem.fit()` | 1 gradient | **Default.** Diagonal Fisher whitening and SGD with Nesterov momentum (lr 1, 0.9). Strictly first-order; the curvature is computed once. Robust from any start in the basin. |
| 2 | `problem.fit(optimiser="lbfgs")` | ~1 gradient | **Usually fastest.** Converged from every benchmark start, and stops on its own once converged. Add `precondition="full"` when you have up to a few hundred parameters. |
| 3 | `problem.fit(precondition="full")` | 1 gradient | **Already near the minimum** (a re-fit, a refined model). A natural-gradient (Newton) step; converges in 1–3 steps there. Unreliable from far away. |

The [benchmarks](fitting_benchmark.md) have the numbers behind these recommendations. In short:

- **L-BFGS** needed the fewest gradients per fit almost everywhere.
- **The default** is several times slower when parameters are strongly correlated, but never
  failed.
- **Natural gradient** is unbeatable near the minimum and fails from far away.

!!! tip "Which preconditioner?"
    `precondition="diagonal"` (the default) only needs the diagonal of the Fisher matrix, so it
    scales to many parameters (e.g. pixel OPD maps). `precondition="full"` needs the whole
    matrix, which costs about n gradient evaluations and n² memory, and it also captures
    correlations. Below a few hundred parameters the full matrix is usually affordable, and
    in the benchmarks it was even cheaper to compute than the diagonal.

## The likelihood

`FitProblem(model, params, data, nll, *nll_args)` takes any negative log likelihood with the
signature `nll(prediction, data, *nll_args, **nll_kwargs) -> scalar`. By default the prediction
is `model.set(params).model()`; pass `predict=lambda params, model: ...` for anything else,
e.g. several exposures stacked into one array.

- **Sum, don't average.** Use a proper negative log likelihood, summed over pixels. The Fisher
  matrix, the uncertainties and the convergence test are all in its units, and a mean over
  pixels would make the uncertainties too large by √(number of pixels).
- **Mask NaN pixels before arithmetic.** NaN in `data` marks excluded pixels. `np.nansum` gives
  NaN *gradients* at those pixels (JAX computes 0 × NaN = NaN), so mask with `np.where` first.
  The built-in `gaussian_nll(prediction, data, std)` does this.
- **Any likelihood works.** Correlated noise (e.g. a row-correlated Gaussian) and Poisson
  noise are both fine. The Fisher matrix is computed from the likelihood you supply.

```python
def poisson_nll(prediction, data):
    ok = ~np.isnan(data)
    return np.sum(np.where(ok, prediction - np.where(ok, data, 0.0) * np.log(prediction), 0.0))

problem = FitProblem(telescope, params, data, poisson_nll)
```

## Choosing what to fit

`params` is a dictionary of `{path: array}`. `trainable` selects entries. `True`/`False`
fits or freezes a whole key, and a boolean array selects individual entries. Keys you don't
mention are fitted. Frozen entries keep their starting values exactly.

```python
problem = FitProblem(telescope, params, data, gaussian_nll, std,
                     trainable={"aberrations.coefficients": np.arange(8) < 4})  # first 4 only
problem.active_labels   # ['aberrations.coefficients[0]', ..., 'source.position[1]']
```

## Running and resuming

`run(n)` takes up to `n` steps and returns the fit, so calls chain. Everything is recorded
before each step:

```python
fit = problem.fit()
fit.run(200).run(200)        # resumes; optimiser state is kept
fit.losses                   # loss before each step
fit.active_history           # (n_steps, n_active), columns named by fit.active_labels
fit.param_history            # {key: (n_steps, *shape)}
fit.moves                    # step sizes, in conditional sigma
plt.plot(fit.active_history, label=fit.active_labels)
```

`fit.refresh()` recomputes the curvature where the fit is now, with the same settings, and
restarts the optimiser from there. History is kept. It helps when the starting point was far
off and the curvature has changed, e.g. `problem.fit(precondition="full").run(5).refresh()`.

### Stopping on convergence

`run` stops early once two consecutive steps each move every parameter by less than `tol`
conditional sigma. It then sets `fit.converged = True`.

- **Default:** `tol = 0.01` for `optimiser="lbfgs"`, and off otherwise. In the benchmarks,
  L-BFGS stopped within 0.01σ of the minimum from every start, a few steps after the loss had
  converged.
- **Normalisation:** because the test is in sigma, it doesn't depend on the absolute value of
  the loss. It only assumes the loss is a proper negative log likelihood.
- **Other optimisers:** you can set `tol` for them too, e.g.
  `problem.fit(precondition="full", tol=0.01)`. Small steps reliably mean convergence for
  L-BFGS and natural gradient, whose steps approach the full remaining distance. Diagonal SGD
  on correlated parameters can take small steps while still far away, so use a smaller `tol`
  there.
- **Turning it off:** `tol=0` never stops early. Avoid running L-BFGS far past convergence
  regardless: at the exact minimum its line search struggles, and steps cost about 10× more.

## Uncertainties

Compute the full curvature at the best fit:

```python
c = problem.curvature(fit.params, precondition="full")
c.std               # marginal 1-sigma, in your dict shapes (frozen entries = 0)
c.covariance        # (n, n), ordered by c.labels
c.correlation       # (n, n) correlation matrix
c.conditional_std   # 1/sqrt(F_ii): 1-sigma with everything else fixed (optimistic)
c.condition_number  # degeneracy after removing units; large = near-degenerate parameters
```

- **Default kind:** `kind="expected"` (the default) is the Fisher matrix. That gives the
  Cramér–Rao bound, and it's exact for fixed-(co)variance Gaussian and for Poisson noise.
- **Observed information:** `kind="observed"` uses the actual Hessian of the loss against the
  data, an accepted alternative at the best fit.
- **Frozen parameters** are treated as known, not marginalised over.

## Options in detail

### Curvature

`problem.curvature(params=None, *, precondition, kind, damping, curvature_data, batch_size)`:

| Option | Values | Meaning |
|---|---|---|
| `precondition` | `"diagonal"` (default), `"full"` | Compute the diagonal only (enough to whiten; no covariances), or the whole matrix. |
| `kind` | `"expected"` (default), `"observed"` | The Fisher matrix, computed with the data replaced by the model prediction (always positive semi-definite), or the actual Hessian of the loss (can be indefinite away from the minimum; always computes the full matrix). |
| `damping` | float ≥ 0, `"full"` only | Adds `damping` × each diagonal entry before whitening, for near-singular problems. Covariances stay undamped. |
| `curvature_data` | array, `"observed"` only | Data to differentiate against instead of the problem's data. |
| `batch_size` | int | How many derivative columns at once; memory only. |

### Optimiser

`problem.fit(curvature=None, optimiser=None, tol=None, **curvature_options)`:

- **`None`** (default) picks the right SGD for the preconditioner: Nesterov momentum for
  diagonal, plain natural gradient for full. Momentum is not used with full whitening, where
  the curvature is already the identity: it only adds overshoot, and it failed from far starts
  in the benchmarks.
- **`"lbfgs"`** is `optax.lbfgs()`, with a line search, so there's no learning rate. It reuses
  the line search's value and gradient, so a step costs about one gradient.
- **Any optax transform** works and receives gradients in whitened coordinates. For example,
  `optax.sgd(1.0)` gives plain SGD, and `optax.sgd(0.7, 0.9, nesterov=True)` is the default
  with a wider stability margin, if a default fit oscillates. **Avoid `optax.adam`:** its
  per-parameter rescaling undoes the whitening, and it never converged from far starts in the
  benchmarks.

## How it works

The fit optimises `u`, where the active parameters are `x = x0 + T u`. `T` is chosen so that the
curvature in `u` at the start point `x0` is (close to) the identity. All linear algebra uses the
unit-free normalised Fisher matrix `F_n = S F S`, with `S = diag(1/sqrt(diag F))`; see
[unit-free Hessians](curvature.md).

- **Diagonal:** `T = S / sqrt(λ_max(F_n))`. `λ_max` is estimated by power iteration (about 20
  Fisher-vector products), so a step with lr 1 is a Newton step along the stiffest direction,
  and plain gradient descent is stable for lr < 2. Correlated directions converge at about
  `λ_min/λ_max` per step, which momentum largely fixes. Nesterov momentum 0.9 at lr 1 is
  stable while the whitened curvature stays below ~1.36.
- **Full:** `T = S L⁻ᵀ`, where `L Lᵀ = F_n + damping·I`. Gradient descent in `u` is then
  natural-gradient descent with the Fisher matrix frozen at `x0`: lr 1 is a Newton
  (Fisher-scoring) step.
- **Expected Fisher:** `F = Jᵀ H J`, where `J` is the Jacobian of the prediction and `H` the
  Hessian of the likelihood with respect to the prediction. It's evaluated with the data
  replaced by the prediction, so it's exact for any likelihood whose score vanishes there
  (fixed-covariance Gaussian, Poisson) and always positive semi-definite.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ValueError: No positive curvature (freeze these parameters)` | Those parameters don't affect the prediction at this point. Freeze them with `trainable`, or move the start (e.g. even Zernikes at exactly zero aberration have no first-order effect). |
| `ValueError: Normalised curvature is singular or indefinite` (full) | Some combination of parameters is (nearly) degenerate. Freeze one of them, use the diagonal preconditioner, or set `damping=0.01`. |
| `ValueError: Nonfinite curvature` | Usually NaN gradients from NaN pixels: mask with `np.where` in your `nll`. |
| `FloatingPointError: Nonfinite loss/gradient` | The step diverged and no update was applied. Lower the learning rate, use L-BFGS, or `refresh()`. |
| The default fit oscillates | Pass `optimiser=optax.sgd(0.7, 0.9, nesterov=True)`, or use L-BFGS. |
| A fit is slow | Use L-BFGS, or `precondition="full"` if n is modest. Check `problem.curvature(..., precondition="full").condition_number`: large values mean genuinely correlated parameters. |
| It converges to the wrong answer | No method here escapes a wrong local minimum. Use several starts, fit coarse-to-fine (low-order terms first), or enumerate known ambiguities (e.g. the even-Zernike sign flip). |

## Performance notes

- **Steps cost about one gradient.** That's true of every method here, including L-BFGS,
  whose line search usually accepts its first trial step. The model evaluation dominates; the
  optimiser's own work is negligible.
- **A compiled loop barely helps.** Running the whole solve inside one compiled JAX loop (as
  optimistix does) saved only 2–9% over the Python loop here; see the
  [benchmarks](fitting_benchmark.md#compiled-loops-and-optimistix). The Python loop keeps
  history, progress bars and resumable `run(n)`.
- **Whitening matters more than the library.** Run on raw dLux parameters, optimistix's BFGS
  failed to converge. In the whitened coordinates, it matched this module's L-BFGS. To use
  another optimiser library, minimise
  `lambda u: problem.loss(c.active + c.transform @ u)` with `c = problem.curvature(...)`.
- **Float64 is required** (`jax_enable_x64`). The whitening itself would work in float32, but
  line searches and convergence tests need loss differences that float32 can't resolve for
  large images.
