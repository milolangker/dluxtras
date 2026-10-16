import numpy as np
import pytest

from dluxtras.throughput import (
    bin_throughput,
    filter_integrate,
    fractional_bandwidth,
    load_throughput,
    spectrum_args,
)


def test_sparse_table_integral_and_units(tmp_path):
    path = tmp_path / "bench_throughput.dat"
    path.write_text("# wavelength [Angstrom] throughput\n40000 0\n40500 1\n41000 0\n")
    table = load_throughput(path, wavelength_unit="angstrom")
    np.testing.assert_allclose(table[:, 0], [4e-6, 4.05e-6, 4.1e-6])
    _, area = filter_integrate(table[:, 0], table[:, 1], 20)
    assert area.sum() == pytest.approx(5e-8)
    waves, mean = bin_throughput(table[:, 0], table[:, 1], 20)
    assert waves.shape == (20,) and np.isfinite(mean).all()
    # Mean throughput per bin averages to (integral / band width) = 0.5.
    assert mean.mean() == pytest.approx(0.5)
    assert waves[0] == pytest.approx(4e-6 + 2.5e-9)


def test_spectrum_args_from_arrays():
    from dluxtras.spectra import CombinedBasisSpectrum

    wavelengths = np.linspace(600e-9, 700e-9, 101)
    throughput = 0.5 * np.exp(-0.5 * ((wavelengths - 650e-9) / 15e-9) ** 2)
    args = spectrum_args(wavelengths, throughput, 12, photons=1e6, n_coarse=4)
    assert set(args) == {"wavels", "filt_weights", "basis_weights", "basis", "n_coarse"}
    assert args["basis_weights"].sum() == pytest.approx(1e6)  # incident photons
    spectrum = CombinedBasisSpectrum(**args)
    detected = float(spectrum.proper_flux())
    assert 0 < detected < 1e6  # detected = incident x throughput
    assert detected == pytest.approx(np.sum(args["filt_weights"]) * 1e6 / 12)
    # Default 1% threshold trims the Gaussian wings.
    assert args["wavels"][0] > 600e-9 and args["wavels"][-1] < 700e-9
    with pytest.raises(ValueError, match="basis_weights"):
        spectrum_args(wavelengths, throughput, 12, basis=np.ones((12, 1)))


def test_fractional_bandwidth_top_hat():
    wavelengths = np.linspace(500e-9, 800e-9, 301)
    throughput = ((wavelengths >= 600e-9) & (wavelengths <= 700e-9)).astype(float)
    bw, centre = fractional_bandwidth(wavelengths, throughput)
    # One neighbouring sample (1 nm) is kept on each side of the band.
    assert centre == pytest.approx(650e-9)
    assert bw == pytest.approx(102e-9 / 650e-9)


def test_invalid_inputs(tmp_path):
    path = tmp_path / "filter.dat"
    path.write_text("1 0\n2 0\n")
    with pytest.raises(ValueError, match="zero"):
        load_throughput(path)
    with pytest.raises(ValueError, match="wavelength_unit"):
        load_throughput(path, wavelength_unit="furlong")
    with pytest.raises(ValueError, match="increasing"):
        bin_throughput(np.array([2e-6, 1e-6]), np.ones(2), 4)
    with pytest.raises(ValueError, match="threshold"):
        bin_throughput(np.array([1e-6, 2e-6]), np.ones(2), 4, threshold=1.5)
    with pytest.raises(ValueError, match="nwavels"):
        filter_integrate(np.array([1e-6, 2e-6]), np.ones(2), 0)


def test_spectrum_args_rejects_ignored_photons_and_integer_k():
    from dluxtras.spectra import recommend_wavelength_sampling

    wavelengths = np.linspace(600e-9, 700e-9, 101)
    throughput = np.ones(101)
    assert spectrum_args(wavelengths, throughput, 4)["basis_weights"].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="photons"):
        spectrum_args(wavelengths, throughput, 4, photons=1e6, basis_weights=np.ones(4))
    with pytest.raises(ValueError, match="photons"):
        spectrum_args(wavelengths, throughput, 4, photons=1e6, basis=np.eye(4),
                      basis_weights=np.ones(4))
    assert recommend_wavelength_sampling(3.0, 1e9, verbose=False)["speedup"] > 0  # 3.0 is fine
    with pytest.raises(ValueError, match="integer"):
        recommend_wavelength_sampling(2.5, 1e9, verbose=False)
