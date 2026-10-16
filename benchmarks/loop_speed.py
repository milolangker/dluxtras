"""Does compiling the whole solve into one JAX loop speed up L-BFGS? And optimistix?

    pip install optimistix      # optional: the optimistix rows are skipped without it
    python benchmarks/loop_speed.py

Same problem as fitting_benchmark.py (30 Zernikes + position + flux, n = 33), from the
"physical" start, on a cheap model and on a ~13x heavier one (256-px pupil, 10
wavelengths). All rows minimise the same Fisher-whitened loss, except the last, which
hands optimistix the raw parameters (metres, radians, photons).

Rows:
- FisherFit: dluxtras' Python loop (one jitted step per call, history, checks), with
  its default stop on convergence (tol = 0.01 sigma).
- Bare Python loop / compiled loop: the same jitted L-BFGS step, for the number of steps
  FisherFit needed to reach a loss within 0.01 of the minimum, run either from Python or
  inside one lax.fori_loop.
- optimistix: optx.minimise, which compiles the whole solve, with its own stopping rule.

Times exclude compilation. Final errors are in marginal sigma at the truth.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)

import dLux as dl  # noqa: E402
import dLux.utils as dlu  # noqa: E402
import jax.numpy as np  # noqa: E402
import numpy as onp  # noqa: E402
import optax  # noqa: E402

from dluxtras.fitting import FitProblem, gaussian_nll  # noqa: E402

try:
    import optimistix as optx
except ImportError:
    optx = None


def make_problem(wf_npixels, psf_npixels, n_wavelengths, n_zernikes=30, seed=0):
    js = np.arange(4, 4 + n_zernikes)
    basis = np.asarray(dlu.zernike_basis(js, dlu.pixel_coords(wf_npixels, 1.0), 1.0))
    rng = onp.random.default_rng(seed)
    coefficients = np.asarray(rng.normal(0, 30e-9, n_zernikes))
    optics = dl.AngularOpticalSystem(
        wf_npixels, 1.0, [("aperture", dl.CircularAperture(0.5, normalise=True)),
                          ("aberrations", dl.BasisOptic(basis, coefficients=coefficients))],
        psf_npixels, 0.1, 2)
    position = np.array([3e-8, -2e-8])
    telescope = dl.Telescope(optics, dl.PointSource(
        wavelengths=np.linspace(0.9e-6, 1.1e-6, n_wavelengths), position=position, flux=1e6))
    data = telescope.model()
    truth = {"aberrations.coefficients": coefficients, "source.position": position,
             "source.flux": np.array(1e6)}
    start = {"aberrations.coefficients": coefficients + np.asarray(rng.normal(0, 10e-9, n_zernikes)),
             "source.position": position + np.array([1.7e-7, 1.2e-7]),
             "source.flux": np.array(1.3e6)}
    problem = FitProblem(telescope, start, data, gaussian_nll, np.sqrt(data + 10.0))
    return problem, truth, start


def block(x):
    return jax.block_until_ready(x)


def timed(fn):
    """Seconds for fn(), after one untimed call to compile."""
    block(fn())
    t = time.perf_counter()
    out = block(fn())
    return time.perf_counter() - t, out


def main():
    for label, args in [("cheap: 64-px pupil, 3 wavelengths", (64, 32, 3)),
                        ("heavy: 256-px pupil, 10 wavelengths", (256, 64, 10))]:
        problem, truth, start = make_problem(*args)
        x_truth = problem.selection.active_of(truth)
        loss_min = float(problem.loss(x_truth))
        sigma = np.sqrt(np.diag(problem.curvature(truth, precondition="full").covariance))
        c = problem.curvature(start, precondition="full")
        n = c.active.size

        def whitened(u):
            return problem.loss(c.active + c.transform @ u)

        grad = jax.jit(jax.value_and_grad(whitened))
        seconds, _ = timed(lambda: grad(np.zeros(n)))
        print(f"\n### {label}: n = {n}, one value_and_grad = {seconds * 1e3:.1f} ms\n")
        print("| method | steps | time | final error |")
        print("|---|---|---|---|")

        def error(x):
            return f"{float(np.max(np.abs(x - x_truth) / sigma)):.1g} sigma"

        # dluxtras FisherFit, timed step by step (the first step includes compilation).
        fit = problem.fit(c, optimiser="lbfgs")
        times, k = [], None
        while not fit.converged and fit.steps < 300:
            t = time.perf_counter()
            fit.step()
            times.append(time.perf_counter() - t)
            if k is None and fit.losses[-1] - loss_min < 0.01:
                k = fit.steps - 1
        per_step = float(onp.median(times[1:]))
        print(f"| FisherFit L-BFGS (stops at tol 0.01) | {fit.steps} | "
              f"{per_step * fit.steps * 1e3:.0f} ms | {error(fit.active)} |")

        # The same jitted step for k steps: Python loop vs one compiled loop.
        opt = optax.lbfgs()
        value_and_grad = optax.value_and_grad_from_state(whitened)

        @jax.jit
        def step(u, state):
            value, g = value_and_grad(u, state=state)
            updates, state = opt.update(g, state, u, value=value, grad=g, value_fn=whitened)
            return optax.apply_updates(u, updates), state

        def python_loop():
            u = np.zeros(n)
            state = opt.init(u)
            for _ in range(k):
                u, state = step(u, state)
            return u

        @jax.jit
        def compiled_loop():
            u = np.zeros(n)
            return jax.lax.fori_loop(0, k, lambda i, us: step(*us), (u, opt.init(u)))[0]

        for name, fn in [("bare jitted step, Python loop", python_loop),
                         ("same step, whole loop compiled", compiled_loop)]:
            seconds, u = timed(fn)
            print(f"| {name} | {k} | {seconds * 1e3:.0f} ms | "
                  f"{error(c.active + c.transform @ u)} |")

        if optx is None:
            print("| optimistix | not installed | | |")
            continue

        def optimistix_row(name, fn, y0, solver, to_x):
            def solve():
                sol = optx.minimise(lambda y, args: fn(y), solver, y0, max_steps=3000,
                                    throw=False)
                return sol.value, sol.stats["num_steps"]

            seconds, (y, nsteps) = timed(jax.jit(solve))
            print(f"| {name} | {int(nsteps)} | {seconds * 1e3:.0f} ms | {error(to_x(y))} |")

        def unwhiten(u):
            return c.active + c.transform @ u

        for tol in (1e-4, 1e-8):
            optimistix_row(f"optimistix BFGS, whitened, rtol=atol={tol:g}", whitened,
                           np.zeros(n), optx.BFGS(rtol=tol, atol=tol), unwhiten)
        optimistix_row("optimistix LBFGS, whitened, rtol=atol=1e-8", whitened, np.zeros(n),
                       optx.LBFGS(rtol=1e-8, atol=1e-8), unwhiten)
        optimistix_row("optimistix BFGS, raw units, rtol=1e-8, atol=1e-14", problem.loss,
                       c.active, optx.BFGS(rtol=1e-8, atol=1e-14), lambda x: x)


if __name__ == "__main__":
    main()
