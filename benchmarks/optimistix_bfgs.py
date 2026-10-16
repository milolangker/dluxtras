"""Would optimistix's BFGS make a better default than optax's L-BFGS inside FisherFit?

    pip install optimistix
    python benchmarks/optimistix_bfgs.py      # ~5 min

FisherFit takes one step per call (for history, progress bars and resumable `run(n)`),
so this drives `optimistix.BFGS` the same way: `solver.init`, then a compiled
"step until the line search accepts" per call, with FisherFit's convergence rule (two
consecutive steps under 0.01 conditional sigma). It is compared with
`FisherFit(optimiser="lbfgs")` on the problems and starts of fitting_benchmark.py.

Cells: worst case over the starts at each distance of steps to the stop / steps until
the loss was within 0.01 of the minimum / final error in marginal sigma.

Result (2026-10-16, optimistix 0.1.0): with full whitening optimistix took slightly
fewer steps (e.g. 21 vs 28 from the physical start, n = 33); with diagonal whitening it
took up to 2x more (66 vs 33). Per-step cost was the same. The driving below relies on
the solvers' init/step methods and on closure conversion, and optimistix's LBFGS state
cannot be carried between compiled calls this way, so FisherFit keeps optax's L-BFGS.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as np  # noqa: E402
import numpy as onp  # noqa: E402
import optimistix as optx  # noqa: E402

from fitting_benchmark import make_problem, starts  # noqa: E402


def optimistix_fit(problem, curvature, solver, tol=0.01, cap=300):
    """Steps, per-step seconds, final parameters and losses of a stepwise optimistix fit."""
    T, x0, scales = curvature.transform, curvature.active, curvature.scales

    def apply(u):
        return T * u if T.ndim == 1 else T @ u

    u = np.zeros_like(x0)
    fn = eqx.filter_closure_convert(lambda u, args: (problem.loss(x0 + apply(u)), None), u, None)
    f_struct, aux_struct = fn.out_struct
    state = solver.init(fn, u, None, {}, f_struct, aux_struct, frozenset())

    @eqx.filter_jit
    def until_accepted(u, state):
        n0 = state.num_accepted_steps

        def cond(carry):
            return (carry[1].num_accepted_steps == n0) & (carry[2] < 30)

        def body(carry):
            u, state, i = carry
            u, state, _ = solver.step(fn, u, None, {}, state, frozenset())
            return u, state, i + 1

        return jax.lax.while_loop(cond, body, (u, state, 0))[:2]

    u, state = until_accepted(u, state)  # first accepted step only evaluates at the start
    small, steps, times, losses = 0, 0, [], []
    while steps < cap:
        t = time.perf_counter()
        u_new, state = until_accepted(u, state)
        losses.append(float(state.f_info.f))
        times.append(time.perf_counter() - t)
        move = float(np.max(np.abs(apply(u_new - u)) / scales))
        u, steps = u_new, steps + 1
        small = small + 1 if move < tol else 0
        if small >= 2:
            break
    return steps, float(onp.median(times[1:])), x0 + apply(u), losses


def main():
    for n_zernikes in (8, 30):
        problem, truth, physical, _ = make_problem(n_zernikes)
        start_sets, reference = starts(problem, truth, physical, (3, 30, 300), 3)
        x_truth = problem.selection.active_of(truth)
        sigma = np.sqrt(np.diag(reference.covariance))
        loss_min = float(problem.loss(x_truth))
        print(f"\n### n = {reference.scales.size}\n")
        print("| whitening | method | ms/step | " + " | ".join(start_sets) + " |")
        print("|---|---|---|" + "---|" * len(start_sets))
        for precondition in ("diagonal", "full"):
            rows = {"optax L-BFGS (FisherFit)": [], "optimistix BFGS": []}
            ms = {key: [] for key in rows}
            for name, params_list in start_sets.items():
                cells = {key: [] for key in rows}
                for params in params_list:
                    c = problem.curvature(params, precondition=precondition)
                    fit = problem.fit(c, optimiser="lbfgs")
                    times = []
                    while not fit.converged and fit.steps < 300:
                        t = time.perf_counter()
                        fit.step()
                        times.append(time.perf_counter() - t)
                    hit = onp.flatnonzero(onp.array(fit.losses) - loss_min < 0.01)
                    err = float(np.max(np.abs(fit.active - x_truth) / sigma))
                    cells["optax L-BFGS (FisherFit)"].append(
                        (fit.steps, hit[0] if hit.size else -1, err))
                    ms["optax L-BFGS (FisherFit)"].append(onp.median(times[1:]))
                    steps, per_step, x, losses = optimistix_fit(
                        problem, c, optx.BFGS(rtol=1e-30, atol=1e-30))
                    hit = onp.flatnonzero(onp.array(losses) - loss_min < 0.01)
                    err = float(np.max(np.abs(x - x_truth) / sigma))
                    cells["optimistix BFGS"].append((steps, hit[0] + 1 if hit.size else -1, err))
                    ms["optimistix BFGS"].append(per_step)
                for key, values in cells.items():
                    worst = [max(v[i] for v in values) for i in range(3)]
                    rows[key].append(f"{worst[0]} / {worst[1]} / {worst[2]:.1g}")
            for key, cells in rows.items():
                print(f"| {precondition} | {key} | {onp.median(ms[key]) * 1e3:.2f} | "
                      + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
