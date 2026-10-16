"""Wavefront eigenmodes and marginalisation, checked against dense joint Fisher matrices."""

import dLux as dl
import dLux.utils as dlu
import jax
import jax.numpy as np
import numpy as onp
import pytest

from dluxtras.eigenmodes import leading_modes
from dluxtras.wavefront_eigenmodes import (
    MaskedPixelOPD,
    mode_maps,
    nuisance_shapes,
    wavefront_eigenmodes,
    wavefront_fisher,
)

KEY = "optics.fit_opd.coefficients"
POSITIONS = ([0.0, 0.0], [2e-7, -1e-7])  # radians


def make_models():
    transmission = onp.asarray(dlu.circle(dlu.pixel_coords(16, 1.0), 0.5), dtype=float)
    optics = dl.AngularOpticalSystem(
        16, 1.0, [("aperture", dl.TransmissiveLayer(transmission, normalise=True))], 8, 0.1, 2)
    rng = onp.random.default_rng(1)
    layer = MaskedPixelOPD(transmission)
    layer = layer.set("coefficients", np.asarray(20.0 * rng.normal(size=layer.coefficients.size)))
    optics = optics.insert_layer(("fit_opd", layer), 1)
    models = [dl.Telescope(optics, dl.PointSource(wavelengths=np.array([1e-6]), flux=1e5,
                                                  position=np.array(p))) for p in POSITIONS]
    return models, layer


def dense_marginalised(models, layer, per_model_keys, per_model_dirs, shared_keys):
    """Brute force: joint Fisher over (q, every nuisance), then its Schur complement."""
    q0 = onp.asarray(layer.coefficients)
    n = q0.size
    blocks, eta_sizes = [], []
    for model in models:
        def predict_q(q, model=model):
            return model.set(KEY, q).model().ravel()

        Jq = onp.asarray(jax.jacfwd(predict_q)(np.asarray(q0)))

        def keyed(keys, model=model):
            if not keys:
                return onp.zeros((Jq.shape[0], 0))
            cols = []
            for key in keys:
                value = np.asarray(model.get(key), dtype=float)
                f = lambda v, key=key, shape=value.shape: model.set(key, v.reshape(shape)).model().ravel()
                cols.append(onp.asarray(jax.jacfwd(f)(value.ravel())).reshape(Jq.shape[0], -1))
            return onp.concatenate(cols, axis=1)

        own = onp.concatenate([keyed(per_model_keys), Jq @ per_model_dirs], axis=1)
        shared = keyed(shared_keys)
        mu = onp.asarray(model.model()).ravel()
        blocks.append((Jq, own, shared, 1 / mu))
        eta_sizes.append(own.shape[1])
    m_shared = blocks[0][2].shape[1]
    n_eta = sum(eta_sizes) + m_shared
    F = onp.zeros((n + n_eta, n + n_eta))
    start = n
    for (Jq, own, shared, w), size in zip(blocks, eta_sizes):
        J = onp.zeros((Jq.shape[0], n + n_eta))
        J[:, :n] = Jq
        J[:, start:start + size] = own
        J[:, n + sum(eta_sizes):] = shared
        F += J.T @ (w[:, None] * J)
        start += size
    Fqq, Fqn, Fnn = F[:n, :n], F[:n, n:], F[n:, n:]
    d = onp.sqrt(onp.diag(Fnn))
    inv = onp.linalg.pinv(Fnn / onp.outer(d, d), rcond=1e-10, hermitian=True) / onp.outer(d, d)
    return Fqq - Fqn @ inv @ Fqn.T, Fqq


def test_unmarginalised_and_marginalised_fisher_match_dense():
    models, layer = make_models()
    D = nuisance_shapes(layer, ["defocus"])
    dense, plain = dense_marginalised(models, layer, ["source.position"], D,
                                      ["optics.psf_pixel_scale"])
    F0 = wavefront_fisher(models, KEY).dense()
    onp.testing.assert_allclose(F0, plain, rtol=1e-7, atol=1e-10 * onp.abs(plain).max())
    F = wavefront_fisher(models, KEY, marginalise=["source.position", D],
                         marginalise_shared=["optics.psf_pixel_scale"]).dense()
    onp.testing.assert_allclose(F, dense, rtol=1e-6, atol=1e-8 * onp.abs(dense).max())
    # Marginalising only removes information.
    assert onp.linalg.eigvalsh(plain - F).min() > -1e-8 * onp.abs(plain).max()
    # Each model's defocus is marginalised, so defocus is in the null space.
    assert onp.linalg.norm(F @ D) < 1e-8 * onp.linalg.norm(plain @ D)


def test_eigenmodes_dense_vs_lanczos_piston_and_maps():
    models, layer = make_models()
    modes = wavefront_eigenmodes(models, KEY, k=4, marginalise=["source.position"])
    assert modes.layer is not None and onp.all(modes.residuals < 1e-5)
    onp.testing.assert_allclose(modes.vectors.sum(axis=0), 0, atol=1e-8)  # piston removed
    dense = leading_modes(modes.fisher.dense(), 4, project_out=onp.ones(modes.reference.size))
    onp.testing.assert_allclose(modes.values, dense.values, rtol=1e-6)
    maps = modes.maps()
    assert maps.shape == (4, 16, 16) and onp.isnan(maps[:, 0, 0]).all()
    onp.testing.assert_allclose(maps, mode_maps(layer, modes.vectors))
    assert modes.sigma.shape == (4,) and modes.coefficients(modes.reference).shape == (4,)


def test_masks_and_variances_per_model():
    models, layer = make_models()
    mask = onp.zeros((16, 16), dtype=bool)
    mask[0, :] = True
    one = wavefront_fisher([models[0]], KEY, bad_masks=[mask]).dense()  # one-model list
    same = wavefront_fisher(models[0], KEY, bad_masks=mask).dense()
    onp.testing.assert_allclose(one, same)
    both = wavefront_fisher(models, KEY, bad_masks=[mask, None], noise="gaussian",
                            variance=[1.0, 4.0]).dense()
    parts = (wavefront_fisher(models[0], KEY, bad_masks=mask, noise="gaussian").dense()
             + wavefront_fisher(models[1], KEY, noise="gaussian", variance=4.0).dense())
    onp.testing.assert_allclose(both, parts, rtol=1e-10)
    with pytest.raises(ValueError, match="per model"):
        wavefront_fisher(models, KEY, bad_masks=[mask])
