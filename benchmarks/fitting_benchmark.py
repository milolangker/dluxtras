"""Benchmark FisherFit methods on a high-signal dLux point-source fit.

    python benchmarks/fitting_benchmark.py            # full run (~10 min on a laptop CPU)
    python benchmarks/fitting_benchmark.py --quick    # one start per distance, smaller cap

Model: 64-pixel pupil, Zernike aberrations (metres), source position (radians)
and flux (~1e6 detected photons), 3 wavelengths, Gaussian noise with variance
data + 10. Data are noiseless, so the truth is the exact minimum.

Starts are drawn at k marginal sigma from the truth along random correlated
directions (offset = k * chol(covariance) @ z / sqrt(n), so the loss excess is
roughly k^2 / 2), plus one "physical" start: 10 nm rms coefficient errors,
40 mas position offset and 30% flux error.

Reported per method: steps until the loss is within 0.01 of the minimum (the
median over starts, with the number of starts that converged), and the wall
time to get there = curvature setup + steps x (time per step after JIT
warm-up). Compile time of the step function (~1 s, once per fit) is excluded.
Output is a markdown table on stdout.
"""

import argparse
import dataclasses
import time

import jax

jax.config.update("jax_enable_x64", True)

import dLux as dl  # noqa: E402
import dLux.utils as dlu  # noqa: E402
import jax.numpy as np  # noqa: E402
import numpy as onp  # noqa: E402
import optax  # noqa: E402

from dluxtras.fitting import FitProblem, gaussian_nll  # noqa: E402

TOLERANCE = 0.01  # loss excess counted as converged


def make_problem(n_zernikes, seed=0):
    js = np.arange(4, 4 + n_zernikes)
    basis = np.asarray(dlu.zernike_basis(js, dlu.pixel_coords(64, 1.0), 1.0))
    rng = onp.random.default_rng(seed)
    coefficients = np.asarray(rng.normal(0, 30e-9, n_zernikes))
    optics = dl.AngularOpticalSystem(
        64, 1.0, [("aperture", dl.CircularAperture(0.5, normalise=True)),
                  ("aberrations", dl.BasisOptic(basis, coefficients=coefficients))], 32, 0.1, 2)
    position = np.array([3e-8, -2e-8])
    telescope = dl.Telescope(optics, dl.PointSource(
        wavelengths=np.linspace(0.9e-6, 1.1e-6, 3), position=position, flux=1e6))
    data = telescope.model()
    truth = {"aberrations.coefficients": coefficients, "source.position": position,
             "source.flux": np.array(1e6)}
    physical = {"aberrations.coefficients": coefficients + np.asarray(rng.normal(0, 10e-9, n_zernikes)),
                "source.position": position + np.array([1.7e-7, 1.2e-7]),
                "source.flux": np.array(1.3e6)}
    problem = FitProblem(telescope, physical, data, gaussian_nll, np.sqrt(data + 10.0))
    return problem, truth, physical, float(data.sum())


def starts(problem, truth, physical, distances, n_random, seed=1):
    reference = problem.curvature(truth, precondition="full")
    root = np.linalg.cholesky(reference.covariance)
    x_truth = problem.selection.active_of(truth)
    rng = onp.random.default_rng(seed)
    out = {}
    for k in distances:
        draws = [rng.normal(size=x_truth.size) for _ in range(n_random)]
        out[f"{k} sigma"] = [problem.selection.expand(x_truth + k * root @ np.asarray(z)
                                                       / onp.sqrt(x_truth.size)) for z in draws]
    out["physical"] = [physical]
    return out, reference


def old_default(curvature_diag, curvature_full):
    """spectrum's full_hessian=True + normalise_step=True: diag(F^-1), lambda_max-normalised."""
    d = np.sqrt(np.diag(curvature_full.covariance))
    m = d[:, None] * curvature_full.matrix * d[None, :]
    return dataclasses.replace(curvature_diag, transform=d / np.sqrt(np.linalg.eigvalsh(m)[-1]))


METHODS = [
    # label, precondition, optimiser, refresh_at
    ("diagonal SGD, lr 1.0 (default)", "diagonal", 1.0, None),
    ("diagonal SGD, lr 0.5", "diagonal", 0.5, None),
    ("diagonal SGD, Nesterov (lr 1, momentum 0.9)", "diagonal", "nesterov", None),
    ("diagonal SGD, heavy-ball momentum 0.9", "diagonal", optax.sgd(1.0, 0.9), None),
    ("diagonal Adam, lr 0.1 (not recommended)", "diagonal", optax.adam(0.1), None),
    ("full SGD, lr 0.5", "full", 0.5, None),
    ("full SGD, lr 1.0", "full", 1.0, None),
    ("full SGD, lr 1.0, refresh at step 5", "full", 1.0, 5),
    ("diagonal L-BFGS", "diagonal", "lbfgs", None),
    ("full L-BFGS", "full", "lbfgs", None),
    ("baseline: old diag(F^-1) SGD, lr 0.5", "old", 0.5, None),
]


def timed_curvature(problem, params, precondition):
    t = time.perf_counter()
    curvature = problem.curvature(params, precondition=precondition)
    return curvature, time.perf_counter() - t


def step_time(problem, curvature, optimiser, nsteps=20):
    fit = problem.fit(curvature, optimiser=optimiser).run(3, progress=False)  # compile
    t = time.perf_counter()
    fit.run(nsteps, progress=False)
    return (time.perf_counter() - t) / nsteps


def steps_to_converge(problem, curvature, optimiser, refresh_at, cap, loss_min):
    fit = problem.fit(curvature, optimiser=optimiser)
    extra = 0.0
    try:
        if refresh_at:
            fit.run(refresh_at, progress=False)
            t = time.perf_counter()
            fit.refresh()
            extra = time.perf_counter() - t
        fit.run(cap - fit.steps, progress=False)
    except FloatingPointError:
        return None, extra
    excess = onp.array(fit.losses + [float(problem.loss(fit.active))]) - loss_min
    hit = onp.flatnonzero(excess < TOLERANCE)
    return (int(hit[0]) if hit.size else None), extra


def benchmark(n_zernikes, distances, n_random, sgd_cap, lbfgs_cap):
    problem, truth, physical, photons = make_problem(n_zernikes)
    start_sets, reference = starts(problem, truth, physical, distances, n_random)
    loss_min = float(problem.loss(problem.selection.active_of(truth)))
    n = reference.scales.size
    print(f"\n### n = {n} parameters ({n_zernikes} Zernikes + position + flux), "
          f"{photons:.2g} detected photons, cond(F_n) at truth = {reference.condition_number:.3g}\n")

    # Per-step cost after warm-up, measured once at the physical start.
    timed_curvature(problem, physical, "diagonal")  # first-call tracing overheads
    cd, td = timed_curvature(problem, physical, "diagonal")
    cf, tf = timed_curvature(problem, physical, "full")
    setup = {"diagonal": td, "full": tf, "old": td + tf}
    per_step = {}
    for label, precondition, optimiser, _ in METHODS:
        curvature = {"diagonal": cd, "full": cf, "old": old_default(cd, cf)}[precondition]
        per_step[label] = step_time(problem, curvature, optimiser)
    grad = per_step[METHODS[0][0]]  # an SGD step is one value_and_grad
    print(f"Curvature setup (incl. compile): diagonal {td:.2f} s (~{td / grad:.0f} gradients), "
          f"full {tf:.2f} s (~{tf / grad:.0f} gradients). One value_and_grad ~ "
          f"{grad * 1e3:.1f} ms.\n")

    header = "| method | ms/step | " + " | ".join(
        f"{name}: steps (conv.) / time" for name in start_sets) + " |"
    print(header)
    print("|" + "---|" * (2 + len(start_sets)))
    results = {label: [] for label, *_ in METHODS}
    for name, params_list in start_sets.items():
        for params in params_list:
            cd = problem.curvature(params)
            cf = problem.curvature(params, precondition="full")
            for label, precondition, optimiser, refresh_at in METHODS:
                curvature = {"diagonal": cd, "full": cf, "old": old_default(cd, cf)}[precondition]
                cap = lbfgs_cap if optimiser == "lbfgs" else sgd_cap
                steps, extra = steps_to_converge(problem, curvature, optimiser, refresh_at, cap,
                                                 loss_min)
                results[label].append((name, steps, extra))
    for label, precondition, optimiser, _ in METHODS:
        cells = []
        for name, params_list in start_sets.items():
            runs = [(s, e) for nm, s, e in results[label] if nm == name]
            done = [s for s, _ in runs if s is not None]
            if not done:
                cells.append(f"not converged (0/{len(runs)})")
                continue
            median = float(onp.median(done))
            seconds = setup[precondition] + median * per_step[label] + max(e for _, e in runs)
            cells.append(f"{median:.0f} ({len(done)}/{len(runs)}) / {seconds:.2f} s")
        print(f"| {label} | {per_step[label] * 1e3:.1f} | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_random = 1 if args.quick else 3
    sgd_cap, lbfgs_cap = (1000, 100) if args.quick else (3000, 300)
    print(f"Converged = loss within {TOLERANCE} of the minimum. Cells: median steps "
          f"(starts converged / starts) / wall time = setup + steps x ms/step. "
          f"Caps: SGD {sgd_cap}, L-BFGS {lbfgs_cap} steps.")
    for n_zernikes in (8, 30):
        benchmark(n_zernikes, (3, 30, 300), n_random, sgd_cap, lbfgs_cap)


if __name__ == "__main__":
    main()
