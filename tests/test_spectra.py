"""Tests for spectra, Hermite wavelength-sparse propagation and point sources."""

import dLux as dl
import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

jax.config.update("jax_enable_x64", True)

from dluxtras.sources import HermitePointSource, PointSourceArcsec
from dluxtras.spectra import (
    CombinedBasisSpectrum,
    hermite_node_coeffs,
    propagate_spectrum,
)

N_WAV = 12
WAVELENGTHS = np.linspace(2.0e-6, 2.4e-6, N_WAV)


def make_optics():
    return dl.AngularOpticalSystem(
        wf_npixels=32,
        diameter=1.0,
        layers=[("aperture", dl.CircularAperture(0.5))],
        psf_npixels=16,
        psf_pixel_scale=0.05,
        oversample=2,
    )


def make_spectrum(n_coarse=None, n=N_WAV):
    wavels = np.linspace(2.0e-6, 2.4e-6, n)
    return CombinedBasisSpectrum(
        wavels,
        np.exp(-(((wavels - 2.2e-6) / 0.2e-6) ** 2)),
        np.linspace(1.0, 2.0, n) * 100.0,
        np.eye(n),
        n_coarse=n_coarse,
    )


def test_hermite_cubic_exact():
    fine = np.linspace(0.0, 1.0, 40)
    coarse = np.linspace(0.0, 1.0, 5)
    w = np.linspace(0.5, 1.5, 40)
    def f(x):
        return 2 * x**3 - x**2 + 0.5 * x + 3
    def df(x):
        return 6 * x**2 - 2 * x + 0.5
    cv, cd = hermite_node_coeffs(fine, w, coarse)
    approx = np.sum(cv * f(coarse)) + np.sum(cd * df(coarse))
    assert np.allclose(approx, np.sum(w * f(fine)), rtol=1e-12)
    assert np.allclose(cv.sum(), w.sum())


def test_hermite_coeffs_nonuniform_nodes():
    fine = np.linspace(1.0, 3.0, 25)
    coarse = np.array([1.0, 1.3, 2.0, 2.2, 3.0])
    w = np.ones(25)
    def f(x):
        return x**3 - 4 * x
    def df(x):
        return 3 * x**2 - 4
    cv, cd = hermite_node_coeffs(fine, w, coarse)
    assert np.allclose(np.sum(cv * f(coarse) + cd * df(coarse)), np.sum(f(fine)), rtol=1e-12)


def test_exact_when_n_coarse_none():
    optics, spec = make_optics(), make_spectrum()
    pos = np.array([1e-7, -2e-7])
    a = propagate_spectrum(optics, spec, pos, 3.0)
    b = optics.propagate(spec.wavelengths, pos, spec.weights * 3.0)
    assert np.array_equal(a, b)


def test_full_nodes_matches_exact():
    optics, pos = make_optics(), np.array([1e-7, -2e-7])
    exact = propagate_spectrum(optics, make_spectrum(None), pos)
    herm = propagate_spectrum(optics, make_spectrum(N_WAV), pos)
    assert np.abs(herm - exact).max() / exact.max() < 1e-12


def test_small_n_coarse_close_not_identical():
    optics, pos = make_optics(), np.array([1e-7, -2e-7])
    exact = propagate_spectrum(optics, make_spectrum(None), pos)
    herm = propagate_spectrum(optics, make_spectrum(4), pos)
    err = np.abs(herm - exact).max() / exact.max()
    assert 0 < err < 1e-2
    assert np.allclose(herm.sum(), exact.sum(), rtol=1e-3)


def test_return_psf():
    optics, pos = make_optics(), np.array([0.0, 0.0])
    psf = propagate_spectrum(optics, make_spectrum(4), pos, return_psf=True)
    ref = optics.propagate(WAVELENGTHS, pos, make_spectrum().weights, return_psf=True)
    assert isinstance(psf, dl.PSF)
    assert psf.data.shape == (32, 32)  # psf_npixels * oversample
    assert np.allclose(psf.pixel_scale, ref.pixel_scale)


def test_return_wf_raises():
    with pytest.raises(ValueError):
        propagate_spectrum(make_optics(), make_spectrum(4), np.zeros(2), return_wf=True)


@pytest.mark.parametrize("bad", [1, 0, N_WAV + 1, 3.5, "a"])
def test_invalid_n_coarse(bad):
    with pytest.raises((ValueError, TypeError)) as exc:
        make_spectrum(bad)
    if not isinstance(bad, str):
        assert isinstance(exc.value, ValueError)


def test_gradient_basis_weights():
    optics, pos = make_optics(), np.array([1e-7, 0.0])
    spec = make_spectrum(4)

    @eqx.filter_jit
    def loss(bw):
        s = eqx.tree_at(lambda s: s.basis_weights, spec, bw)
        return np.sum(propagate_spectrum(optics, s, pos) ** 2)

    g = jax.grad(loss)(spec.basis_weights)
    assert g.shape == spec.basis_weights.shape
    assert np.all(np.isfinite(g)) and np.any(g != 0)
    # finite-difference check on one coefficient
    e = np.zeros(N_WAV).at[5].set(1e-3)
    fd = (loss(spec.basis_weights + e) - loss(spec.basis_weights - e)) / 2e-3
    assert np.allclose(fd, g[5], rtol=1e-5)


def test_hermite_point_source_matches_pointsource():
    optics, spec = make_optics(), make_spectrum(None)
    pos = np.array([1e-7, 2e-7])
    ref = dl.PointSource(spectrum=spec, position=pos, flux=5.0).model(optics)
    out = HermitePointSource(spectrum=spec, position=pos, flux=5.0).model(optics)
    assert np.allclose(out, ref, rtol=1e-12, atol=0)


def test_point_source_arcsec_filter_jit():
    optics = make_optics()
    tel = dl.Telescope(
        optics=optics,
        source=PointSourceArcsec(spectrum=make_spectrum(4), position=np.array([0.01, -0.02])),
    )
    out = eqx.filter_jit(lambda t: t.model())(tel)
    assert out.shape == (32, 32)  # psf_npixels * oversample and np.all(np.isfinite(out))
    exact = dl.Telescope(
        optics=optics,
        source=PointSourceArcsec(spectrum=make_spectrum(None), position=np.array([0.01, -0.02])),
    ).model()
    assert np.abs(out - exact).max() / exact.max() < 1e-2


def test_low_memory_matches_vmap():
    from dluxtras.spectra import CombinedBasisSpectrum, propagate_spectrum

    optics = dl.AngularOpticalSystem(32, 1.0, [dl.CircularAperture(0.5)], 16, 0.05, 2)
    wavels = np.linspace(1e-6, 1.5e-6, 12)
    for n_coarse in (None, 4):
        spectra = [
            CombinedBasisSpectrum(wavels, np.ones(12), np.ones(12), np.eye(12),
                                  n_coarse=n_coarse, low_memory=lm)
            for lm in (False, True)
        ]
        a, b = (propagate_spectrum(optics, s, np.array([1e-7, 0.0]), 2.0) for s in spectra)
        assert np.allclose(a, b, rtol=1e-12, atol=1e-15)
        psf = propagate_spectrum(optics, spectra[1], np.zeros(2), return_psf=True)
        assert isinstance(psf, dl.PSF)


def test_load_throughput_clips_tiny_negatives(tmp_path):
    from dluxtras.throughput import load_throughput

    file = tmp_path / "f.txt"
    file.write_text("1.0 0.0\n1.1 -0.0003\n1.2 1.0\n1.3 0.0\n")
    assert load_throughput(file)[1, 1] == 0.0
    file.write_text("1.0 0.0\n1.1 -0.1\n1.2 1.0\n1.3 0.0\n")
    with pytest.raises(ValueError):
        load_throughput(file)


def test_recommend_wavelength_sampling_from_k():
    from dluxtras.spectra import recommend_wavelength_sampling

    small = recommend_wavelength_sampling(4, 1e10, verbose=False)
    assert small["sparse"] and small["n_coarse"] == 8 and small["n_fine"] == 300
    big = recommend_wavelength_sampling(50, 1e9, verbose=False)
    assert not big["sparse"] and big["n_coarse"] is None and big["n_fine"] >= 100


def test_spectrum_projection_and_flux_conventions():
    spectrum = CombinedBasisSpectrum(
        np.array([4e-6, 4.1e-6]),
        np.array([0.5, 1.0]),
        np.array([1.0, 2.0]),
        np.eye(2),
        proj=np.diag(np.array([2.0, 3.0])),
    )
    onp.testing.assert_allclose(spectrum.spec_weights(), [2.0, 6.0])
    assert float(spectrum.flux) == pytest.approx(8.0)  # incident
    assert float(spectrum.proper_flux()) == pytest.approx(7.0)  # detected
