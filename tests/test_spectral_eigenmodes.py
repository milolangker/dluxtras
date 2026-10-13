"""Spectral Fisher matrix, eigenmodes and the constrainable-mode criterion."""

import dLux as dl
import jax
import jax.numpy as np
import numpy as onp
import pytest

from dluxtras.spectral_eigenmodes import (
    attach_spectrum,
    constrainable_modes,
    mode_criterion,
    spectral_eigenmodes,
    spectral_fisher,
)

N_WAV = 6


def make_optics(oversample=2):
    return dl.AngularOpticalSystem(
        16, 1.0, [("aperture", dl.CircularAperture(0.5, normalise=True))], 8, 0.1, oversample
    )


def make_detector(oversample=2):
    return dl.LayeredDetector([("downsample", dl.layers.Downsample(oversample))])


def spectral_args(photons=1e6):
    wavels = onp.linspace(0.8e-6, 1.2e-6, N_WAV)
    return dict(wavels=wavels, filt_weights=onp.linspace(0.5, 1.0, N_WAV),
                basis_weights=onp.full(N_WAV, photons / N_WAV), basis=onp.eye(N_WAV))


def test_spectral_fisher_matches_dense_and_criterion():
    detector = make_detector()
    model = attach_spectrum(make_optics(), spectral_args(), detector=detector,
                            position_arcsec=(0.03, -0.02))
    F = onp.asarray(spectral_fisher(model, noise="poisson"))

    def image(bin_flux):
        return model.set("source.bin_flux", bin_flux).model()

    q0 = model.source.bin_flux
    J = onp.asarray(jax.jacfwd(image)(q0)).reshape(-1, N_WAV)
    mu = onp.asarray(image(q0)).ravel()
    onp.testing.assert_allclose(F, J.T @ (J / mu[:, None]), rtol=1e-8)
    # Poisson reference: mu.sum() is the detected photon count N.
    assert mu.sum() == pytest.approx(float(np.sum(q0 * model.source.filt_weights)), rel=0.1)

    modes = spectral_eigenmodes(make_optics(), spectral_args(), k=N_WAV,
                                detector=detector, position_arcsec=(0.03, -0.02))
    assert onp.all(onp.diff(onp.asarray(modes.values)) <= 0)
    onp.testing.assert_allclose(
        onp.asarray(modes.vectors.T @ modes.vectors), onp.eye(N_WAV), atol=1e-10)
    criterion = mode_criterion(modes)
    K = constrainable_modes(modes)
    assert 1 <= K < N_WAV and K == int(onp.sum(criterion < 0.1))
    # 100x more photons: criterion improves by 10x (N^-1/2).
    bright = spectral_eigenmodes(make_optics(), spectral_args(1e8), k=N_WAV,
                                 detector=detector, position_arcsec=(0.03, -0.02))
    finite = onp.isfinite(criterion)
    onp.testing.assert_allclose(mode_criterion(bright)[finite], criterion[finite] / 10, rtol=1e-6)
