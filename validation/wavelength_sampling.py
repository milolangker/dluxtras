"""Check the wavelength-sampling rules of `recommend_wavelength_sampling` on generic optics.

    python validation/wavelength_sampling.py            # full study, ~6 min on an M5 Pro CPU
    python validation/wavelength_sampling.py --quick    # reduced study, ~30 s

`dluxtras.spectra.recommend_wavelength_sampling(K, N)` encodes two empirical rules for how
many wavelengths a broadband point-source model needs, so that the model error stays below
Delta chi^2 = 1 under Poisson noise:

- Hermite sparsity: n_coarse = max(3, 2K) coarse nodes (on a fixed fine grid);
- exact propagation: n_exact = ceil(23.5 K^0.553 (N/1e9)^0.208) wavelength bins;

where K is the number of constrainable spectral modes (mode i counts iff
1/(|a0| sqrt(d_i)) < 0.1, from the spectral Fisher matrix in incident-photons-per-bin
coordinates with Poisson variance = image) and N is the number of DETECTED photons in the
stamp. The rules were fitted on JWST/NIRCam models (the private `k_vs_fov` study notebook).
This script re-derives the measurements with generic optics and synthetic filters, using only
dluxtras, dLux, JAX and NumPy, and asks whether either rule ever undershoots.

Method (as in the notebook), for every (pupil, filter, stamp):

1. Truth: `NFINE` exact monochromatic PSFs on the `spectrum_args` grid of the filter
   (flat incident spectrum), each propagated at oversample 3 and summed to detector pixels;
   the truth image is `CombinedBasisSpectrum(**args).weights @ psfs`. It is checked against
   `dl.Telescope(optics, HermitePointSource(...), detector)` in every case.
2. K: the unit-flux Fisher F_1 = J^T diag(1/mu_1) J with J[:, j] = filt_j psf_j is built
   once from the cached PSFs, decomposed with `spectral_eigenmodes.leading_modes`, and K(N)
   follows analytically (the criterion scales as N^-1/2): `constrainable_modes` is applied
   to a `SpectralModes` rescaled to N detected photons. F_1 is cross-checked against
   `spectral_fisher` once per pupil.
3. Delta chi^2(model) = N sum_p (I_model - I_truth)^2 / I_truth, both images divided by the
   truth image sum, so that N is the detected photon count in the stamp.
4. Minimum Hermite n_coarse: Hermite images on the same NFINE-bin spectrum (so pure
   interpolation error) for every integer n_coarse from 3 upwards, using the cached PSFs and
   their wavelength derivatives (one JVP per fine wavelength) with
   `spectra.hermite_node_coeffs`. This is the same arithmetic as `propagate_spectrum`, and
   is checked against `HermitePointSource` at the measured minimum in every case. Reported
   is the smallest n_coarse beyond which every scanned value passes.
5. Minimum exact n: images from `spectrum_args(..., nwavels=n)` (their own binning, so this
   is the quadrature error) on a grid of n, refined to the integer by bisection.
6. The two rules (taken from `recommend_wavelength_sampling` itself) are evaluated at K(N)
   and their images are tested directly: a rule undershoots if Delta chi^2 >= 1 at its
   prediction.

Differences from the notebook:

- Optics: `dl.AngularOpticalSystem` with two generic pupils instead of the JWST pupil and a
  measured OPD: "clear", an unobscured, unaberrated circle; and "obscured", a 20% central
  obscuration, four 1%-of-D spider vanes and 50 nm RMS of random Zernike aberration
  (Noll 4-11, about lambda_c/20, Strehl ~0.9). D = 1 m, 256-pixel pupil, oversample 3.
- Filters: four synthetic smooth top-hats (tanh edges, a 20% linear throughput tilt, like a
  QE slope) centred on 1 um, with the fractional bandwidths (at 1% of peak, as
  `fractional_bandwidth`) of the notebook's F212N, F182M, F444W and F322W2
  (0.030, 0.166, 0.292, 0.532), and per-filter pixel scales giving the same lambda_c/D in
  detector pixels (2.14, 1.86, 2.20, 1.62). Stamps 32, 64, 128 and 256 px; the source sits
  at the same sub-pixel offset (0.3, -0.2) px; N = 1e8, 1e9, 1e10.
- Minima are resolved to the integer (every n_coarse is scanned, n_exact is bisected)
  rather than on the notebook's coarse grids, and the rule predictions are checked directly.
- The full run keeps the notebook's 600 fine bins. `--quick` uses only the obscured pupil,
  the M17 and W53 filters, 32 and 64 px stamps and 300 fine bins.

Runtime on an Apple M5 Pro (CPU, float64): about 6 minutes for the full study (each
monochromatic PSF costs only a few ms with a 256-pixel pupil, and the Hermite scan reuses
cached PSF derivatives), and about 30 s with `--quick`. Progress goes to stderr; the
markdown summary to stdout.
"""

import argparse
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import dLux as dl  # noqa: E402
import dLux.utils as dlu  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as np  # noqa: E402
import numpy as onp  # noqa: E402

from dluxtras import (  # noqa: E402
    CombinedBasisSpectrum,
    HermitePointSource,
    recommend_wavelength_sampling,
    spectrum_args,
)
from dluxtras.spectra import coarse_indices, hermite_node_coeffs  # noqa: E402
from dluxtras.spectral_eigenmodes import (  # noqa: E402
    SpectralModes,
    constrainable_modes,
    leading_modes,
    spectral_eigenmodes,
)
from dluxtras.throughput import fractional_bandwidth  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------

DIAMETER = 1.0  # m
WF_NPIX = 256  # pupil sampling
OS = 3  # oversampling; images are summed to detector pixels
LAM_C = 1e-6  # m, centre of every synthetic filter
OFF_PIX = onp.array([0.3, -0.2])  # sub-pixel source offset, detector pixels
NS = (1e8, 1e9, 1e10)  # detected photons in the stamp
THRESHOLD = 0.1  # constrainable-mode criterion
CHUNK = 25  # wavelengths per vmapped propagation

# name: (fractional bandwidth at 1% of peak, lambda_c / D in detector pixels); the values are
# those of NIRCam F212N, F182M, F444W and F322W2 in the notebook.
FILTERS = {"N3": (0.030, 2.14), "M17": (0.166, 1.86), "W29": (0.292, 2.20), "W53": (0.532, 1.62)}
STAMPS = (32, 64, 128, 256)
PUPILS = ("clear", "obscured")
NFINE = 600

# exact-propagation grid (refined by bisection)
NEX_GRID = (5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400)


def log(message):
    """Print a progress message to stderr.

    Parameters
    ----------
    message : str
        Message to print.
    """
    print(message, file=sys.stderr, flush=True)


def filter_curve(frac_bw, lam_c=LAM_C, edge=0.04, tilt=0.2, n=8001):
    """Synthetic filter throughput: a smooth top-hat with tanh edges and a linear tilt.

    The 1%-of-peak band is `lam_c * (1 -/+ frac_bw / 2)`; the edges are tanh steps of width
    `edge` times the half-power width, and the throughput rises linearly by `tilt` across
    the band (peak about 0.5).

    Parameters
    ----------
    frac_bw : float
        Fractional bandwidth at 1% of peak throughput.
    lam_c : float, optional
        Central wavelength in metres. Default `LAM_C`.
    edge : float, optional
        Edge width as a fraction of the half-power width. Default 0.04.
    tilt : float, optional
        Fractional throughput rise from the blue to the red end. Default 0.2.
    n : int, optional
        Number of samples. Default 8001.

    Returns
    -------
    wavelengths : numpy.ndarray
        Wavelengths in metres, shape (n,).
    throughput : numpy.ndarray
        Dimensionless throughput, shape (n,).
    """
    full = frac_bw * lam_c
    z = 2.2976  # 0.5 (1 + tanh(-z)) = 0.01
    width = full / (1 + 2 * z * edge)  # half-power width
    a, b, w = lam_c - width / 2, lam_c + width / 2, edge * width
    wl = onp.linspace(lam_c - 0.75 * full, lam_c + 0.75 * full, n)
    slope = 1 - tilt / 2 + tilt * (wl - a) / width
    tp = 0.5 * slope * 0.25 * (1 + onp.tanh((wl - a) / w)) * (1 + onp.tanh((b - wl) / w))
    return wl, tp


def make_pupil(kind, npix=WF_NPIX, seed=1):
    """Pupil layer: transmission (and OPD) of a generic circular aperture.

    Parameters
    ----------
    kind : str
        "clear" (unobscured, unaberrated circle) or "obscured" (20% central obscuration,
        four spider vanes 1% of D wide, 50 nm RMS of Noll 4-11 Zernike aberration).
    npix : int, optional
        Pupil sampling. Default `WF_NPIX`.
    seed : int, optional
        Seed of the random aberration. Default 1.

    Returns
    -------
    dl.Optic
        Normalised pupil layer.
    """
    coords = dlu.pixel_coords(npix, DIAMETER)
    if kind == "clear":
        return dl.Optic(dl.CircularAperture(DIAMETER / 2).transmission(coords, DIAMETER / npix),
                        normalise=True)
    aperture = dl.CompoundAperture([
        dl.CircularAperture(DIAMETER / 2),
        dl.CircularAperture(0.2 * DIAMETER / 2, occulting=True),
        dl.Spider(0.01 * DIAMETER, np.array([45.0, 135.0, 225.0, 315.0])),
    ])
    transmission = aperture.transmission(coords, DIAMETER / npix)
    js = np.arange(4, 12)
    coeffs = onp.random.default_rng(seed).normal(size=len(js))
    opd = np.tensordot(np.asarray(coeffs), dlu.zernike_basis(js, coords, DIAMETER), 1)
    opd = opd * 50e-9 / np.sqrt(np.mean(opd[transmission > 0.5] ** 2))
    return dl.Optic(transmission, opd, normalise=True)


def make_optics(pupil, lamd_px, stamp):
    """Angular optical system with lambda_c / D = `lamd_px` detector pixels.

    Parameters
    ----------
    pupil : dl.Optic
        Pupil layer from `make_pupil`.
    lamd_px : float
        lambda_c / D in detector pixels.
    stamp : int
        Stamp width in detector pixels.

    Returns
    -------
    optics : dl.AngularOpticalSystem
        Optics producing an `OS`-times oversampled PSF.
    position : jax.Array
        Source position in radians (the sub-pixel offset `OFF_PIX`).
    """
    scale = float(dlu.rad2arcsec(LAM_C / DIAMETER)) / lamd_px  # arcsec / detector pixel
    optics = dl.AngularOpticalSystem(WF_NPIX, DIAMETER, [("pupil", pupil)], stamp, scale, OS)
    return optics, dlu.arcsec2rad(np.asarray(OFF_PIX * scale))


# ---------------------------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------------------------

@eqx.filter_jit
def _mono(optics, wavels, position):
    return jax.vmap(
        lambda w: dlu.downsample(optics.propagate_mono(w, position), OS, mean=False))(wavels)


@eqx.filter_jit
def _mono_jvp(optics, wavels, position):
    def one(w):
        return jax.jvp(
            lambda x: dlu.downsample(optics.propagate_mono(x, position), OS, mean=False),
            (w,), (np.ones_like(w),))
    return jax.vmap(one)(wavels)


def psf_stack(optics, wavels, position, derivatives=False):
    """Detector-resolution monochromatic PSFs (and d/dlambda), in fixed-size chunks.

    Parameters
    ----------
    optics : dl.AngularOpticalSystem
        Optics to propagate through.
    wavels : array_like
        Wavelengths in metres, shape (n,).
    position : Array
        Source position in radians.
    derivatives : bool, optional
        Also return the wavelength derivatives (per metre). Default False.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        PSFs of shape (n, stamp, stamp), and their derivatives if requested.
    """
    wavels = onp.asarray(wavels, float)
    n = wavels.size
    padded = onp.concatenate([wavels, onp.full(-n % CHUNK, wavels[-1])])
    vals, ders = [], []
    for i in range(0, padded.size, CHUNK):
        chunk = np.asarray(padded[i:i + CHUNK])
        if derivatives:
            v, d = _mono_jvp(optics, chunk, position)
            ders.append(onp.asarray(d))
        else:
            v = _mono(optics, chunk, position)
        vals.append(onp.asarray(v))
    vals = onp.concatenate(vals)[:n]
    return (vals, onp.concatenate(ders)[:n]) if derivatives else vals


DETECTOR = dl.LayeredDetector([("downsample", dl.Downsample(OS))])


@eqx.filter_jit
def telescope_image(optics, spectrum, position):
    """Library model `dl.Telescope(optics, HermitePointSource, detector)`, unit flux.

    Parameters
    ----------
    optics : dl.AngularOpticalSystem
        Optics.
    spectrum : CombinedBasisSpectrum
        Spectrum (its `n_coarse` and `low_memory` are honoured).
    position : Array
        Source position in radians.

    Returns
    -------
    Array
        Detector image.
    """
    source = HermitePointSource(spectrum=spectrum, flux=1.0, position=position)
    return dl.Telescope(optics, source, DETECTOR).model()


# ---------------------------------------------------------------------------------------------
# Fisher, K and Delta chi^2
# ---------------------------------------------------------------------------------------------

def unit_fisher(filt, psfs):
    """Spectral Fisher for unit incident photons per bin, Poisson variance = image.

    Parameters
    ----------
    filt : numpy.ndarray
        Mean throughput per bin, shape (n,).
    psfs : numpy.ndarray
        Detector-resolution PSFs, shape (n, stamp, stamp).

    Returns
    -------
    fisher : numpy.ndarray
        F_1 = J^T diag(1 / mu_1) J, shape (n, n), with J[:, j] = filt_j psf_j.
    n1 : float
        Detected photons of the unit reference, mu_1.sum().
    """
    jac = filt[:, None] * psfs.reshape(len(filt), -1)  # (n, npix)
    mu1 = jac.sum(0)
    assert mu1.min() > 0
    fisher = (jac / mu1) @ jac.T
    return (fisher + fisher.T) / 2, float(mu1.sum())


def eigenpairs(fisher):
    """All eigenvalues (descending) and eigenvectors via `leading_modes`.

    Only the first two outputs of `leading_modes` are used, so this works whether it
    returns `(values, vectors)` or `(values, vectors, residuals)`.

    Parameters
    ----------
    fisher : numpy.ndarray
        Dense Fisher matrix, shape (n, n).

    Returns
    -------
    values : numpy.ndarray
        Eigenvalues, numerically null ones set to zero, shape (n,).
    vectors : numpy.ndarray
        Unit-norm eigenvectors as columns, shape (n, n).
    """
    values, vectors = tuple(leading_modes(fisher, fisher.shape[0]))[:2]
    return onp.asarray(values), onp.asarray(vectors)


def k_of_n(case, photons):
    """Number of constrainable modes at `photons` detected photons.

    Rescales the unit-flux eigen-decomposition to reference q0 = (N / N_1) per bin
    (F_N = F_1 N_1 / N) and applies `spectral_eigenmodes.constrainable_modes`.

    Parameters
    ----------
    case : dict
        Case with keys `wavels`, `filt`, `values`, `vectors` and `n1`.
    photons : float
        Detected photons N in the stamp.

    Returns
    -------
    int
        K(N).
    """
    s = photons / case["n1"]
    modes = SpectralModes(case["wavels"], case["filt"], onp.full(case["filt"].size, s),
                          None, case["values"] / s, case["vectors"])
    return constrainable_modes(modes, THRESHOLD)


def dchi2_per_photon(image, truth, total):
    """Delta chi^2 per detected photon of `image` against `truth`.

    Parameters
    ----------
    image, truth : numpy.ndarray
        Unit-flux images (any shape, same size).
    total : float
        Truth image sum.

    Returns
    -------
    float
        sum (image/total - truth/total)^2 / (truth/total); multiply by N for Delta chi^2.
    """
    image, truth = onp.ravel(image) / total, onp.ravel(truth) / total
    return float(onp.sum((image - truth) ** 2 / truth))


def rules(K, photons):
    """Predictions of `recommend_wavelength_sampling`.

    Parameters
    ----------
    K : int
        Number of constrainable modes (>= 1).
    photons : float
        Detected photons.

    Returns
    -------
    n_coarse, n_exact : int
        `max(3, 2K)` and `ceil(23.5 K^0.553 (N/1e9)^0.208)`.
    """
    sparse = recommend_wavelength_sampling(K, photons, min_speedup=0.0, verbose=False)
    exact = recommend_wavelength_sampling(K, photons, min_speedup=onp.inf, verbose=False)
    return sparse["n_coarse"], exact["n_fine"]


# ---------------------------------------------------------------------------------------------
# One (pupil, filter, stamp) case
# ---------------------------------------------------------------------------------------------

def run_case(pupil_kind, pupil, fname, stamp, nfine, curves):
    """Truth, K(N), minimum Hermite n_coarse and minimum exact n for one case.

    Parameters
    ----------
    pupil_kind : str
        Pupil name.
    pupil : dl.Optic
        Pupil layer.
    fname : str
        Filter name (key of `FILTERS`).
    stamp : int
        Stamp width in detector pixels.
    nfine : int
        Number of fine bins of the truth.
    curves : dict
        Throughput tables, filter name -> (wavelengths, throughput).

    Returns
    -------
    dict
        Results: K, minima, rule predictions and their Delta chi^2, checks.
    """
    t0 = time.time()
    frac, lamd_px = FILTERS[fname]
    wl_tab, tp_tab = curves[fname]
    optics, pos = make_optics(pupil, lamd_px, stamp)
    nmax = max(NS)

    # 1. truth: nfine exact PSFs (with d/dlambda, reused by the Hermite scan)
    args = spectrum_args(wl_tab, tp_tab, nwavels=nfine, photons=1.0)
    spectrum = CombinedBasisSpectrum(**args)
    weights = onp.asarray(spectrum.weights)
    wavels, filt = onp.asarray(args["wavels"]), onp.asarray(args["filt_weights"])
    psfs, dpsfs = psf_stack(optics, wavels, pos, derivatives=True)
    truth = onp.tensordot(weights, psfs, 1)
    total = float(truth.sum())
    lib = onp.asarray(telescope_image(optics, CombinedBasisSpectrum(**args, low_memory=True), pos))
    check_truth = float(onp.abs(lib - truth).max() / truth.max())

    # 2. Fisher once, K(N) analytically
    fisher, n1 = unit_fisher(filt, psfs)
    values, vectors = eigenpairs(fisher)
    case = dict(wavels=wavels, filt=filt, values=values, vectors=vectors, n1=n1)
    K = {N: k_of_n(case, N) for N in NS}

    # 3. Hermite: every integer n_coarse from 3, past the first pass at max(N) and 2K(max N)
    nc_rule = {N: rules(K[N], N)[0] for N in NS}
    chi_h = {}

    def hermite(nc):
        if nc not in chi_h:
            idx = coarse_indices(nfine, nc)
            cv, cd = hermite_node_coeffs(np.asarray(wavels), np.asarray(weights),
                                         np.asarray(wavels[idx]))
            image = (onp.tensordot(onp.asarray(cv), psfs[idx], 1)
                     + onp.tensordot(onp.asarray(cd), dpsfs[idx], 1))
            chi_h[nc] = dchi2_per_photon(image, truth, total)
        return chi_h[nc]

    nc, first_pass = 3, None
    while nc <= nfine:
        if hermite(nc) * nmax < 1 and first_pass is None:
            first_pass = nc
        end = max(int(1.5 * first_pass) + 4, max(nc_rule.values()) + 2) if first_pass else None
        if end is not None and nc >= min(end, nfine):
            break
        nc += 1
    scanned = sorted(chi_h)

    def robust_min(N):
        fails = [n for n in scanned if hermite(n) * N >= 1]
        if fails and fails[-1] == scanned[-1]:
            return None  # never passes within the scan
        return fails[-1] + 1 if fails else scanned[0]

    min_nc = {N: robust_min(N) for N in NS}
    nonmono_h = any(chi_h[a] < chi_h[b] for a, b in zip(scanned[:-1], scanned[1:])
                    if chi_h[a] * nmax >= 1e-3)  # ignore ripples far below threshold
    # cross-check the cached Hermite arithmetic against the library at the minimum (max N)
    nc_chk = min_nc[nmax] or scanned[-1]
    lib_h = onp.asarray(telescope_image(
        optics, CombinedBasisSpectrum(**args, n_coarse=nc_chk, low_memory=True), pos))
    check_hermite = abs(dchi2_per_photon(lib_h, truth, total) / hermite(nc_chk) - 1)
    del psfs, dpsfs

    # 4. exact propagation: grid, then bisection per N; plus the rule predictions directly
    chi_e = {}

    def exact(n):
        if n not in chi_e:
            a = spectrum_args(wl_tab, tp_tab, nwavels=n, photons=1.0)
            w = onp.asarray(CombinedBasisSpectrum(**a).weights)
            chi_e[n] = dchi2_per_photon(
                onp.tensordot(w, psf_stack(optics, a["wavels"], pos), 1), truth, total)
        return chi_e[n]

    for n in NEX_GRID:
        if exact(n) * nmax < 1:
            break
    min_ex = {}
    for N in NS:
        grid = sorted(chi_e)
        passing = [n for n in grid if chi_e[n] * N < 1]
        if not passing:
            min_ex[N] = None
            continue
        hi = passing[0]
        lo = max([n for n in grid if n < hi], default=1)
        while hi - lo > 1:  # chi_e[lo] fails, chi_e[hi] passes
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if exact(mid) * N < 1 else (mid, hi)
        min_ex[N] = hi
    nex_rule = {N: rules(K[N], N)[1] for N in NS}
    chi_rule_ex = {N: exact(nex_rule[N]) * N for N in NS}
    grid = sorted(n for n in chi_e if n in NEX_GRID)
    nonmono_e = any(chi_e[a] < chi_e[b] for a, b in zip(grid[:-1], grid[1:]))

    r_half = stamp / 2 / lamd_px
    result = dict(
        pupil=pupil_kind, filter=fname, stamp=stamp, frac=frac, r_half=r_half, K=K,
        min_nc=min_nc, nc_rule=nc_rule,
        chi_rule_nc={N: hermite(nc_rule[N]) * N for N in NS},
        min_ex=min_ex, nex_rule=nex_rule, chi_rule_ex=chi_rule_ex,
        check_truth=check_truth, check_hermite=check_hermite,
        nonmono_h=nonmono_h, nonmono_e=nonmono_e, seconds=time.time() - t0,
    )
    log(f"{pupil_kind:8s} {fname:4s} {stamp:3d}px  K={list(K.values())}  "
        f"min nc={list(min_nc.values())}  min n_ex={list(min_ex.values())}  "
        f"[{result['seconds']:.0f}s]")
    return result


# ---------------------------------------------------------------------------------------------
# Checks done once per pupil
# ---------------------------------------------------------------------------------------------

def check_library_fisher(pupil, curves, fname, stamp, nbins=60, photons=1e9):
    """Compare the cached-PSF Fisher with `spectral_eigenmodes` (library) on one case.

    Parameters
    ----------
    pupil : dl.Optic
        Pupil layer.
    curves : dict
        Throughput tables.
    fname : str
        Filter name.
    stamp : int
        Stamp width.
    nbins : int, optional
        Number of wavelength bins. Default 60.
    photons : float, optional
        Incident photons of the library reference. Default 1e9.

    Returns
    -------
    rel_diff : float
        max |F_lib - F_mine| / max |F_lib|.
    k_lib, k_mine : int
        K from the library and from the analytic N-scaling at the same detected N.
    """
    optics, pos = make_optics(pupil, FILTERS[fname][1], stamp)
    args = spectrum_args(*curves[fname], nwavels=nbins, photons=photons)
    position_arcsec = OFF_PIX * float(dlu.rad2arcsec(LAM_C / DIAMETER)) / FILTERS[fname][1]
    lib = spectral_eigenmodes(optics, args, k=nbins, detector=DETECTOR, noise="poisson",
                              position_arcsec=position_arcsec)
    filt = onp.asarray(args["filt_weights"])
    fisher, n1 = unit_fisher(filt, psf_stack(optics, args["wavels"], pos))
    q = photons / nbins  # library reference: q incident photons per bin, F_q = F_1 / q
    rel = float(onp.abs(onp.asarray(lib.fisher) - fisher / q).max()
                / onp.abs(onp.asarray(lib.fisher)).max())
    values, vectors = eigenpairs(fisher)
    case = dict(wavels=args["wavels"], filt=filt, values=values, vectors=vectors, n1=n1)
    return rel, constrainable_modes(lib, THRESHOLD), k_of_n(case, q * n1)


def check_truth_convergence(pupil, curves, fname, stamp, nfine):
    """Delta chi^2 per photon between `nfine`-bin and 1.5x-finer truth images.

    Parameters
    ----------
    pupil : dl.Optic
        Pupil layer.
    curves : dict
        Throughput tables.
    fname : str
        Filter name.
    stamp : int
        Stamp width.
    nfine : int
        Number of fine bins of the truth.

    Returns
    -------
    float
        Delta chi^2 per detected photon.
    """
    optics, pos = make_optics(pupil, FILTERS[fname][1], stamp)
    images = []
    for n in (nfine, int(1.5 * nfine)):
        a = spectrum_args(*curves[fname], nwavels=n, photons=1.0)
        w = onp.asarray(CombinedBasisSpectrum(**a).weights)
        images.append(onp.tensordot(w, psf_stack(optics, a["wavels"], pos), 1))
    return dchi2_per_photon(images[1], images[0], float(images[0].sum()))


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

def fmt(x):
    """Format an optional integer ('>' marks a minimum beyond the scan).

    Parameters
    ----------
    x : int or None
        Value.

    Returns
    -------
    str
        The value, or "> scan" for None.
    """
    return "> scan" if x is None else str(x)


def report(results, setup):
    """Print the markdown summary.

    Parameters
    ----------
    results : list of dict
        Output of `run_case`.
    setup : dict
        Run settings and one-off checks.
    """
    out = []
    p = out.append
    p("# Wavelength-sampling rules: generic validation\n")
    p(f"Settings: pupils {', '.join(setup['pupils'])}; filters "
      + ", ".join(f"{f} (frac. bw {setup['frac'][f]:.3f}, lambda_c/D = {FILTERS[f][1]} px)"
                  for f in setup["filters"])
      + f"; stamps {', '.join(map(str, setup['stamps']))} px; {setup['nfine']} fine bins; "
      f"N = {', '.join(f'{N:.0e}' for N in NS)} detected photons.\n")
    p("Checks:\n")
    p(f"- truth image vs `dl.Telescope(HermitePointSource)` (exact): max rel. diff "
      f"{max(r['check_truth'] for r in results):.1e} over all cases")
    p(f"- cached Hermite vs `HermitePointSource(n_coarse)`: max rel. diff of Delta chi^2 "
      f"{max(r['check_hermite'] for r in results):.1e} over all cases")
    for kind, (rel, k_lib, k_mine) in setup["fisher_check"].items():
        p(f"- {kind}: Fisher vs `spectral_eigenmodes` (60 bins): max rel. diff {rel:.1e}; "
          f"K library {k_lib} vs analytic N-scaling {k_mine}")
    for kind, chi in setup["truth_conv"].items():
        p(f"- {kind}: truth convergence ({setup['conv_case']}, {setup['nfine']} vs "
          f"{int(1.5 * setup['nfine'])} bins): Delta chi^2 = {chi * max(NS):.1e} at "
          f"N = {max(NS):.0e}")
    nm_h = [(r["pupil"], r["filter"], r["stamp"]) for r in results if r["nonmono_h"]]
    nm_e = [(r["pupil"], r["filter"], r["stamp"]) for r in results if r["nonmono_e"]]
    p(f"- Delta chi^2 non-monotonic in n_coarse (above 1e-3 at max N) in {len(nm_h)} cases; "
      f"in exact n (on the grid) in {len(nm_e)} cases")
    p("")

    p("## Per-case results\n")
    p("Columns: K(N); measured minimum Hermite n_coarse and the rule max(3, 2K) with its "
      "Delta chi^2; measured minimum exact n and the rule ceil(23.5 K^0.553 (N/1e9)^0.208) "
      "with its Delta chi^2. Delta chi^2 < 1 means the rule is safe.\n")
    for r in results:
        p(f"### {r['pupil']} / {r['filter']} / {r['stamp']} px "
          f"(R_half = {r['r_half']:.1f} lambda_c/D)\n")
        p("| N | K | min n_coarse | 2K rule | Delta chi^2 @ rule | min n_exact | n_exact rule "
          "| Delta chi^2 @ rule |")
        p("|---|---|---|---|---|---|---|---|")
        for N in NS:
            p(f"| {N:.0e} | {r['K'][N]} | {fmt(r['min_nc'][N])} | {r['nc_rule'][N]} | "
              f"{r['chi_rule_nc'][N]:.2g} | {fmt(r['min_ex'][N])} | {r['nex_rule'][N]} | "
              f"{r['chi_rule_ex'][N]:.2g} |")
        p("")

    p("## Summary\n")
    p("Ratios rule / measured minimum over all cases and N (>= 1 means the rule never "
      "undershot).\n")
    p("| subset | cases | min nc/K | max nc/K | 2K rule / min nc: min | median | max "
      "| n_exact rule / min n: min | median | max |")
    p("|---|---|---|---|---|---|---|---|---|---|")

    def row(label, sel):
        rows = [(r, N) for r in results for N in NS if sel(r, N)]
        ok = [(r, N) for r, N in rows if r["min_nc"][N] and r["min_ex"][N]]
        nck = onp.array([r["min_nc"][N] / r["K"][N] for r, N in ok])
        hr = onp.array([r["nc_rule"][N] / r["min_nc"][N] for r, N in ok])
        er = onp.array([r["nex_rule"][N] / r["min_ex"][N] for r, N in ok])
        p(f"| {label} | {len(ok)} | {nck.min():.2f} | {nck.max():.2f} | {hr.min():.2f} | "
          f"{onp.median(hr):.2f} | {hr.max():.2f} | {er.min():.2f} | {onp.median(er):.2f} | "
          f"{er.max():.2f} |")

    row("all", lambda r, N: True)
    for kind in setup["pupils"]:
        row(f"pupil {kind}", lambda r, N, kind=kind: r["pupil"] == kind)
    for f in setup["filters"]:
        row(f"filter {f}", lambda r, N, f=f: r["filter"] == f)
    for N in NS:
        row(f"N = {N:.0e}", lambda r, N, M=N: N == M)
    p("")

    # smallest c such that max(3, ceil(c K)) never undershoots (on a 0.05 grid)
    pairs = [(r["K"][N], r["min_nc"][N]) for r in results for N in NS if r["min_nc"][N]]
    safe_c = next(c for c in onp.arange(0.5, 6.0, 0.05)
                  if all(max(3, onp.ceil(c * k - 1e-9)) >= m for k, m in pairs))
    p(f"Smallest c with max(3, ceil(c K)) >= min n_coarse in every case: c = {safe_c:.2f} "
      "(the rule uses c = 2).\n")

    under_h = [(r, N) for r in results for N in NS if r["chi_rule_nc"][N] >= 1]
    under_e = [(r, N) for r in results for N in NS if r["chi_rule_ex"][N] >= 1]
    unresolved = [(r, N) for r in results for N in NS
                  if r["min_nc"][N] is None or r["min_ex"][N] is None]
    n_tot = len(results) * len(NS)

    def where(items, key):
        return "; ".join(f"{r['pupil']}/{r['filter']}/{r['stamp']}px N={N:.0e} "
                         f"(Delta chi^2 {r[key][N]:.2g})" for r, N in items)

    p("## Verdict\n")
    p(f"- Hermite rule max(3, 2K): undershoots in {len(under_h)} of {n_tot} cases"
      + (f": {where(under_h, 'chi_rule_nc')}" if under_h else "") + ".")
    p(f"- Exact rule ceil(23.5 K^0.553 (N/1e9)^0.208): undershoots in {len(under_e)} of "
      f"{n_tot} cases" + (f": {where(under_e, 'chi_rule_ex')}" if under_e else "") + ".")
    if unresolved:
        p(f"- {len(unresolved)} minima were beyond the scanned range.")
    verdict = "NEITHER rule undershot" if not (under_h or under_e) else "a rule UNDERSHOT"
    p(f"\n**Verdict: {verdict}** (Delta chi^2 < 1 at the rule's prediction in "
      f"{n_tot - len(under_h)} / {n_tot} Hermite and {n_tot - len(under_e)} / {n_tot} "
      "exact cases).")
    p(f"\nTotal runtime {setup['runtime']:.0f} s.")
    print("\n".join(out))


def main():
    """Run the study and print the markdown summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="obscured pupil, 2 filters, 32/64 px stamps, 300 fine bins")
    opts = parser.parse_args()
    t0 = time.time()

    pupils, filters, stamps, nfine = PUPILS, list(FILTERS), STAMPS, NFINE
    if opts.quick:
        pupils, filters, stamps, nfine = ("obscured",), ["M17", "W53"], (32, 64), 300
    curves = {f: filter_curve(FILTERS[f][0]) for f in filters}
    frac = {f: fractional_bandwidth(*curves[f])[0] for f in filters}
    log("fractional bandwidths (1% of peak): "
        + ", ".join(f"{f} {frac[f]:.3f}" for f in filters))

    results, fisher_check, truth_conv = [], {}, {}
    conv_case = (filters[-1], stamps[-1])
    for kind in pupils:
        pupil = make_pupil(kind)
        fisher_check[kind] = check_library_fisher(pupil, curves, filters[-1], stamps[0])
        truth_conv[kind] = check_truth_convergence(pupil, curves, *conv_case, nfine)
        log(f"{kind}: library Fisher check {fisher_check[kind]}, "
            f"truth convergence dchi2/N {truth_conv[kind]:.1e}")
        for f in filters:
            for S in stamps:
                results.append(run_case(kind, pupil, f, S, nfine, curves))

    setup = dict(pupils=pupils, filters=filters, stamps=stamps, nfine=nfine, frac=frac,
                 fisher_check=fisher_check, truth_conv=truth_conv,
                 conv_case=f"{conv_case[0]} {conv_case[1]} px", runtime=time.time() - t0)
    report(results, setup)


if __name__ == "__main__":
    main()
