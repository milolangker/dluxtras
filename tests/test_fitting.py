"""FitProblem / Curvature / FisherFit: exact Fisher, whitening and convergence."""

import dLux as dl
import dLux.utils as dlu
import jax
import jax.numpy as np
import numpy as onp
import optax
import pytest

from dluxtras.fitting import FitProblem, gaussian_nll

RNG = onp.random.default_rng(3)
TEMPLATES = np.asarray(RNG.normal(size=(3, 6, 6)))
LINEAR_TRUTH = {"a": np.array(2.0), "b": np.array([1e-9, -3e-9])}  # mixed units


def linear_predict(p, model):
    return p["a"] * TEMPLATES[0] + 1e9 * (p["b"][0] * TEMPLATES[1] + p["b"][1] * TEMPLATES[2])


def linear_problem(nll=gaussian_nll, *nll_args, start=None, **kwargs):
    """Linear model, so the loss is exactly quadratic. Default noise: std 0.1."""
    data = linear_predict(LINEAR_TRUTH, None)
    start = {"a": np.array(2.5), "b": np.array([3e-9, -1e-9])} if start is None else start
    if nll is gaussian_nll and not nll_args:
        nll_args = (np.full(data.shape, 0.1),)
    return FitProblem(None, start, data, nll, *nll_args, predict=linear_predict, **kwargs)


def dense_jacobian(problem, x):
    return onp.asarray(jax.jacfwd(problem.predict)(x)).reshape(-1, x.size)


def test_expected_fisher_matches_dense_with_nan_pixel():
    std = onp.full((6, 6), 0.1)
    std[2, 3] = onp.nan
    problem = linear_problem(gaussian_nll, np.asarray(std))
    x = problem.selection.initial_active
    J = dense_jacobian(problem, x)
    w = onp.nan_to_num(1 / std.ravel() ** 2)
    F = J.T @ (w[:, None] * J)

    full = problem.curvature(precondition="full")
    onp.testing.assert_allclose(full.matrix, F, rtol=1e-10)
    onp.testing.assert_allclose(full.covariance, onp.linalg.inv(F), rtol=1e-8)
    assert set(full.std) == {"a", "b"} and full.std["b"].shape == (2,)
    onp.testing.assert_allclose(onp.diag(full.normalised), 1.0)

    diagonal = problem.curvature()
    onp.testing.assert_allclose(diagonal.scales, 1 / onp.sqrt(onp.diag(F)), rtol=1e-10)
    onp.testing.assert_allclose(diagonal.lambda_max, full.lambda_max, rtol=1e-6)
    with pytest.raises(ValueError, match="precondition='full'"):
        diagonal.covariance


def test_correlated_and_poisson_fisher():
    # Dense (pixel-correlated) Gaussian: F = J.T A J.
    B = RNG.normal(size=(36, 36))
    A = np.asarray(B @ B.T + 36 * onp.eye(36))

    def correlated_nll(prediction, data):
        r = (data - prediction).ravel()
        return 0.5 * r @ A @ r

    problem = linear_problem(correlated_nll)
    x = problem.selection.initial_active
    J = dense_jacobian(problem, x)
    F = J.T @ onp.asarray(A) @ J
    onp.testing.assert_allclose(problem.curvature(precondition="full").matrix, F, rtol=1e-10)
    onp.testing.assert_allclose(problem.curvature().scales ** -2, onp.diag(F), rtol=1e-10)

    # Poisson: F = J.T diag(1/mu) J, though d2 nll / d data2 = 0.
    def poisson_nll(prediction, data):
        return np.sum(prediction - data * np.log(prediction))

    offset = 50.0
    start = {"a": np.array(2.5), "b": np.array([3e-9, -1e-9])}
    problem = FitProblem(None, start, linear_predict(LINEAR_TRUTH, None) + offset, poisson_nll,
                         predict=lambda p, m: linear_predict(p, m) + offset)
    mu = onp.asarray(problem.predict(x)).ravel()
    J = dense_jacobian(problem, x)
    onp.testing.assert_allclose(problem.curvature().scales ** -2,
                                onp.diag(J.T @ (J / mu[:, None])), rtol=1e-10)


def test_observed_equals_expected_on_noiseless_truth():
    problem = linear_problem(start=LINEAR_TRUTH)
    expected = problem.curvature(precondition="full").matrix
    observed = problem.curvature(precondition="full", kind="observed").matrix
    onp.testing.assert_allclose(observed, expected, rtol=1e-10)


def test_full_whitening_newton_step_and_diagonal_convergence():
    problem = linear_problem()
    truth = problem.selection.active_of(LINEAR_TRUTH)
    # Quadratic loss: natural gradient with lr = 1 lands on the minimum in one step.
    fit = problem.fit(precondition="full").run(1, progress=False)
    onp.testing.assert_allclose(fit.active, truth, rtol=1e-10, atol=1e-20)

    fit = problem.fit(optimiser=optax.sgd(1.0))  # plain SGD, lr 1: monotone without tuning
    fit.run(300, progress=False)
    assert onp.all(onp.diff(fit.losses) <= 1e-12)
    onp.testing.assert_allclose(fit.active, truth, rtol=1e-6)
    assert fit.active_history.shape == (300, 3)
    assert fit.active_labels == ["a", "b[0]", "b[1]"]

    fit = problem.fit().run(300, progress=False)  # default: Nesterov momentum
    onp.testing.assert_allclose(fit.active, truth, rtol=1e-6)


def test_refresh_and_trainable_mask():
    problem = linear_problem(trainable={"b": np.array([True, False])})
    assert problem.active_labels == ["a", "b[0]"]
    fit = problem.fit(optimiser="lbfgs").run(3, progress=False)
    before = fit.active
    fit.refresh()
    onp.testing.assert_allclose(fit.curvature.active, before)
    fit.run(2, progress=False)
    assert len(fit.losses) == 5 and fit.param_history["b"].shape == (5, 2)
    assert float(fit.params["b"][1]) == -1e-9  # frozen entry untouched


def test_degenerate_and_invalid_options():
    def summed(p, model):  # only a + b[0] is constrained
        return (p["a"] + 1e9 * p["b"][0]) * TEMPLATES[0] + 1e9 * p["b"][1] * TEMPLATES[2]

    data = summed(LINEAR_TRUTH, None)
    problem = FitProblem(None, LINEAR_TRUTH, data, gaussian_nll, np.full(data.shape, 0.1),
                         predict=summed)
    with pytest.raises(ValueError, match="singular"):
        problem.curvature(precondition="full")
    assert problem.curvature(precondition="full", damping=0.01).transform.shape == (3, 3)
    problem.curvature()  # the diagonal never needs inverting

    flat = FitProblem(None, LINEAR_TRUTH, data, gaussian_nll, np.full(data.shape, 0.1),
                      predict=lambda p, m: p["a"] * TEMPLATES[0])
    with pytest.raises(ValueError, match="freeze"):
        flat.curvature()

    problem = linear_problem()
    with pytest.raises(ValueError, match="damping"):
        problem.curvature(damping=0.1)
    with pytest.raises(ValueError, match="curvature_data"):
        problem.curvature(curvature_data=problem.data)
    for bad in ("adam", 0.5):
        with pytest.raises(ValueError, match="optax"):
            problem.fit(optimiser=bad)
    with pytest.raises(ValueError, match="not both"):
        problem.fit(problem.curvature(), precondition="full")


def test_dlux_fit_recovers_truth():
    js = np.arange(4, 7)
    basis = np.asarray(dlu.zernike_basis(js, dlu.pixel_coords(16, 1.0), 1.0))
    truth_c = np.array([20e-9, -15e-9, 10e-9])
    optics = dl.AngularOpticalSystem(
        16, 1.0, [("aperture", dl.CircularAperture(0.5, normalise=True)),
                  ("aberrations", dl.BasisOptic(basis, coefficients=truth_c))], 8, 0.1, 2)
    telescope = dl.Telescope(optics, dl.PointSource(wavelengths=np.array([1e-6]), flux=1e5,
                                                    position=np.array([2e-8, -1e-8])))
    data = telescope.model()
    keys = ["aberrations.coefficients", "source.position", "source.flux"]
    truth = {k: telescope.get(k) for k in keys}
    start = {keys[0]: truth_c + 3e-9, keys[1]: np.array([5e-8, 1e-8]), keys[2]: np.array(1.2e5)}
    problem = FitProblem(telescope, start, data, gaussian_nll, np.sqrt(data + 1.0))
    sigma = problem.curvature(truth, precondition="full").std

    fit = problem.fit(optimiser="lbfgs").run(40, progress=False)
    for k in keys:
        assert onp.all(onp.abs(fit.params[k] - truth[k]) < 1e-3 * sigma[k])


def test_lbfgs_stops_on_convergence_by_default():
    problem = linear_problem()
    truth = problem.selection.active_of(LINEAR_TRUTH)
    fit = problem.fit(optimiser="lbfgs").run(300, progress=False)
    assert fit.converged and fit.steps < 50 and len(fit.moves) == fit.steps
    assert fit.moves[-1] < fit.tol and fit.moves[-2] < fit.tol
    onp.testing.assert_allclose(fit.active, truth, rtol=1e-4)
    assert fit.run(10, progress=False).steps == fit.steps  # converged: no more steps
    assert fit.refresh().converged is False  # refresh restarts the test

    assert problem.fit(optimiser="lbfgs", tol=0).run(60, progress=False).steps == 60
    assert problem.fit().tol == 0.0  # SGD: no early stop unless asked
    stopped = problem.fit(precondition="full", tol=0.01).run(100, progress=False)
    assert stopped.converged and stopped.steps < 10
    with pytest.raises(ValueError, match="tol"):
        problem.fit(tol=-1.0)
