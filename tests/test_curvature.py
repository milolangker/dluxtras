"""Unit-free Hessians: exact round trip, covariance, and mixed-unit robustness."""

import jax.numpy as np
import numpy as onp
import pytest

from dluxtras.curvature import correlation, scaled_covariance, scaled_hessian, unscale

# A correlated quadratic loss whose parameters span 18 decades of units.
UNITS = onp.array([1e-9, 1e-9, 1.0, 1e8])
CORR = onp.array([[1.0, 0.6, 0.1, 0.0], [0.6, 1.0, -0.3, 0.2],
                  [0.1, -0.3, 1.0, 0.4], [0.0, 0.2, 0.4, 1.0]])
A = CORR / onp.outer(UNITS, UNITS)  # Hessian in physical units


def loss(x):
    v = np.concatenate([x["coefficients"], x["position"], x["flux"]])
    return 0.5 * v @ np.asarray(A) @ v


X0 = {"coefficients": np.array([3e-8, -2e-8]), "position": np.array([0.5]),
      "flux": np.array([1e9])}


@pytest.mark.parametrize("scales", [None, 2.0, {"coefficients": 1e-9, "position": 1.0,
                                                "flux": 1e8}])
def test_round_trip_and_unit_diagonal(scales):
    normalised, s, unflatten = scaled_hessian(loss, X0, scales)
    # ravel_pytree orders dict leaves by key: coefficients, flux, position.
    order = [0, 1, 3, 2]
    onp.testing.assert_allclose(onp.diag(normalised), 1.0, rtol=1e-12)
    onp.testing.assert_allclose(s, UNITS[order], rtol=1e-12)
    onp.testing.assert_allclose(unscale(normalised, s), A[onp.ix_(order, order)], rtol=1e-10)
    assert set(unflatten(s)) == set(X0)


def test_covariance_and_correlation():
    normalised, s, _ = scaled_hessian(loss, X0)
    cov = onp.linalg.inv(CORR) * onp.outer(UNITS, UNITS)
    order = [0, 1, 3, 2]
    onp.testing.assert_allclose(scaled_covariance(normalised, s), cov[onp.ix_(order, order)],
                                rtol=1e-10)
    d = onp.sqrt(onp.diag(cov))
    onp.testing.assert_allclose(correlation(normalised), (cov / onp.outer(d, d))[
        onp.ix_(order, order)], rtol=1e-10, atol=1e-14)


def test_zero_curvature_keeps_input_scale():
    def flat(x):
        return loss(x) + 0.0 * x["extra"].sum()

    normalised, s, unflatten = scaled_hessian(flat, {**X0, "extra": np.zeros(1)},
                                              {**{k: 1.0 for k in X0}, "extra": 7.0})
    assert float(unflatten(s)["extra"][0]) == 7.0
    assert onp.count_nonzero(onp.diag(normalised) == 0) == 1


def test_rejects_bad_scales():
    with pytest.raises(ValueError):
        scaled_hessian(loss, X0, 0.0)
