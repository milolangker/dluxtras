# Validation: how many wavelengths?

`recommend_wavelength_sampling(K, N)` (see [Spectra](spectra.md#how-many-wavelengths)) turns
the number of constrainable spectral modes `K` and the detected photon count `N` into a
number of wavelengths to propagate:

- **Hermite:** `n_coarse = max(3, 2K)` coarse nodes on a fixed fine grid;
- **Exact:** `n_exact = ceil(23.5 K^0.553 (N/1e9)^0.208)` wavelength bins.

Both rules were fitted to JWST/NIRCam models. This page re-derives them on generic optics
and synthetic filters, using only dluxtras, dLux, JAX and NumPy, and checks whether either
rule ever asks for too few wavelengths.

Reproduce with:

```bash
python validation/wavelength_sampling.py          # ~6 min
python validation/wavelength_sampling.py --quick  # ~30 s, 1 pupil, 2 filters, 2 stamps
```

Run on 2026-10-16, Apple M5 Pro CPU, jax 0.11.2, dLux 0.15.1, NumPy 2.5.4 (full run: 375 s).

## Setup

- **Optics.** `dl.AngularOpticalSystem`, D = 1 m, 256-pixel pupil, oversample 3, summed to
  detector pixels. Two pupils:
    - **clear:** an unobscured, unaberrated circle;
    - **obscured:** a 20% central obscuration, four spider vanes 1% of D wide, and 50 nm RMS
      of random Zernike aberration (Noll 4–11, about λ_c/20, Strehl ~0.9).
- **Filters.** Four synthetic smooth top-hats centred on 1 µm, with tanh edges and a 20%
  linear throughput tilt (like a QE slope). Their fractional bandwidths (at 1% of peak, as
  `fractional_bandwidth` measures them) and λ_c/D in detector pixels match the four NIRCam
  filters of the original study:

    | filter | Δλ/λ_c | λ_c/D [px] | mimics |
    |---|---|---|---|
    | N3 | 0.030 | 2.14 | F212N |
    | M17 | 0.166 | 1.86 | F182M |
    | W29 | 0.291 | 2.20 | F444W |
    | W53 | 0.530 | 1.62 | F322W2 |

- **Stamps** of 32, 64, 128 and 256 px, i.e. half-widths `R_half` of 7–80 λ_c/D. The source
  sits at a sub-pixel offset of (0.3, −0.2) px with a flat incident spectrum.
- **Photons** N = 1e8, 1e9 and 1e10, where N is the **detected** photon count in the stamp
  (the truth image sum).

## Method

For each (pupil, filter, stamp):

1. **Truth.** 600 exact monochromatic PSFs on the `spectrum_args(nwavels=600)` grid, summed
   with `CombinedBasisSpectrum.weights`.
2. **K.** The Fisher matrix in incident-photons-per-bin coordinates, with Poisson variance
   equal to the image, is built once at unit flux from the cached PSFs. `K(N)` then follows
   analytically, since the criterion `1/(|a0| sqrt(d_i))` scales as `N^-1/2`; it's counted
   with `constrainable_modes` (threshold 0.1).
3. **Model error.** `Δχ² = N Σ_p (I_model − I_truth)² / I_truth`, with both images divided by
   the truth sum.
4. **Minimum n_coarse.** Hermite images on the same 600-bin spectrum, so only the
   interpolation error enters. Every integer `n_coarse` from 3 upwards is scanned, and the
   minimum is the smallest value beyond which every scanned value has `Δχ² < 1`.
   (`Δχ²(n_coarse)` isn't monotonic: node alignment makes it ripple, in 18 of 32 cases.)
5. **Minimum n_exact.** Exact images from `spectrum_args(nwavels=n)` with their own binning,
   so this is the quadrature error. Found on a grid, then bisected to the integer.
6. **Rules.** Both rules are evaluated at `K(N)` with `recommend_wavelength_sampling`. Each
   prediction's own image is then tested: a rule undershoots if `Δχ² ≥ 1` there.

The script also runs these consistency checks:

- the truth equals `dl.Telescope(optics, HermitePointSource(...), detector).model()` to 1e-14;
- the cached Hermite images match `HermitePointSource(n_coarse=...)` (Δχ² to 1e-11);
- the Fisher matrix equals `spectral_eigenmodes` to 4e-15, with the same K at the library's
  photon count;
- 600 vs 900 bins in the hardest case (W53, 256 px) gives Δχ² ≤ 1.2e-3 at N = 1e10.

## Results

Each cell gives the values at N = 1e8 / 1e9 / 1e10. `R_half` is in λ_c/D.

**Clear pupil**

| filter | stamp | R_half | K | min n_coarse | max(3, 2K) | min n_exact | n_exact rule |
|---|---|---|---|---|---|---|---|
| N3 | 32 px | 7.5 | 3 / 3 / 4 | 3 / 3 / 4 | 6 / 6 / 8 | 9 / 13 / 24 | 27 / 44 / 82 |
| N3 | 64 px | 15.0 | 4 / 4 / 5 | 4 / 5 / 6 | 8 / 8 / 10 | 11 / 21 / 36 | 32 / 51 / 93 |
| N3 | 128 px | 29.9 | 5 / 5 / 6 | 5 / 6 / 7 | 10 / 10 / 12 | 13 / 25 / 45 | 36 / 58 / 103 |
| N3 | 256 px | 59.8 | 6 / 7 / 8 | 6 / 8 / 10 | 12 / 14 / 16 | 16 / 28 / 50 | 40 / 69 / 120 |
| M17 | 32 px | 8.6 | 6 / 7 / 8 | 6 / 8 / 10 | 12 / 14 / 16 | 22 / 38 / 67 | 40 / 69 / 120 |
| M17 | 64 px | 17.2 | 9 / 11 / 12 | 8 / 12 / 15 | 18 / 22 / 24 | 25 / 44 / 78 | 50 / 89 / 150 |
| M17 | 128 px | 34.4 | 16 / 17 / 19 | 13 / 16 / 21 | 32 / 34 / 38 | 28 / 49 / 86 | 68 / 113 / 194 |
| M17 | 256 px | 68.8 | 28 / 31 / 33 | 21 / 24 / 36 | 56 / 62 / 66 | 29 / 51 / 89 | 92 / 157 / 263 |
| W29 | 32 px | 7.3 | 8 / 9 / 10 | 8 / 11 / 15 | 16 / 18 / 20 | 30 / 52 / 91 | 46 / 80 / 136 |
| W29 | 64 px | 14.5 | 13 / 15 / 16 | 13 / 16 / 21 | 26 / 30 / 32 | 33 / 58 / 103 | 61 / 106 / 176 |
| W29 | 128 px | 29.1 | 23 / 25 / 27 | 20 / 24 / 33 | 46 / 50 / 54 | 35 / 62 / 109 | 83 / 140 / 235 |
| W29 | 256 px | 58.2 | 42 / 46 / 49 | 34 / 43 / 60 | 84 / 92 / 98 | 36 / 63 / 110 | 116 / 196 / 327 |
| W53 | 32 px | 9.9 | 16 / 17 / 19 | 16 / 18 / 23 | 32 / 34 / 38 | 33 / 59 / 103 | 68 / 113 / 194 |
| W53 | 64 px | 19.8 | 28 / 31 / 33 | 26 / 31 / 41 | 56 / 62 / 66 | 36 / 63 / 110 | 92 / 157 / 263 |
| W53 | 128 px | 39.5 | 50 / 58 / 62 | 47 / 59 / 83 | 100 / 116 / 124 | 47 / 63 / 111 | 127 / 222 / 372 |
| W53 | 256 px | 79.0 | 82 / 103 / 116 | 83 / 112 / 145 | 164 / 206 / 232 | 80 / 90 / 111 | 167 / 305 / 526 |

**Obscured, aberrated pupil**

| filter | stamp | R_half | K | min n_coarse | max(3, 2K) | min n_exact | n_exact rule |
|---|---|---|---|---|---|---|---|
| N3 | 32 px | 7.5 | 3 / 3 / 4 | 3 / 3 / 4 | 6 / 6 / 8 | 9 / 13 / 24 | 27 / 44 / 82 |
| N3 | 64 px | 15.0 | 4 / 4 / 5 | 4 / 4 / 5 | 8 / 8 / 10 | 11 / 21 / 36 | 32 / 51 / 93 |
| N3 | 128 px | 29.9 | 5 / 5 / 6 | 5 / 6 / 7 | 10 / 10 / 12 | 15 / 27 / 48 | 36 / 58 / 103 |
| N3 | 256 px | 59.8 | 6 / 7 / 8 | 6 / 8 / 10 | 12 / 14 / 16 | 19 / 32 / 57 | 40 / 69 / 120 |
| M17 | 32 px | 8.6 | 6 / 7 / 8 | 7 / 8 / 10 | 12 / 14 / 16 | 26 / 46 / 81 | 40 / 69 / 120 |
| M17 | 64 px | 17.2 | 9 / 10 / 11 | 8 / 11 / 15 | 18 / 20 / 22 | 30 / 53 / 93 | 50 / 84 / 143 |
| M17 | 128 px | 34.4 | 15 / 17 / 18 | 13 / 16 / 20 | 30 / 34 / 36 | 33 / 59 / 103 | 66 / 113 / 188 |
| M17 | 256 px | 68.8 | 27 / 31 / 33 | 21 / 24 / 33 | 54 / 62 / 66 | 36 / 63 / 110 | 91 / 157 / 263 |
| W29 | 32 px | 7.3 | 8 / 9 / 10 | 8 / 11 / 15 | 16 / 18 / 20 | 34 / 59 / 104 | 46 / 80 / 136 |
| W29 | 64 px | 14.5 | 13 / 14 / 15 | 13 / 16 / 20 | 26 / 28 / 30 | 38 / 67 / 117 | 61 / 102 / 170 |
| W29 | 128 px | 29.1 | 22 / 24 / 26 | 19 / 23 / 31 | 44 / 48 / 52 | 42 / 74 / 128 | 81 / 137 / 230 |
| W29 | 256 px | 58.2 | 40 / 45 / 48 | 31 / 41 / 60 | 80 / 90 / 96 | 43 / 77 / 134 | 112 / 193 / 323 |
| W53 | 32 px | 9.9 | 15 / 17 / 19 | 16 / 18 / 24 | 30 / 34 / 38 | 42 / 74 / 130 | 66 / 113 / 194 |
| W53 | 64 px | 19.8 | 27 / 30 / 32 | 25 / 29 / 41 | 54 / 60 / 64 | 45 / 79 / 138 | 91 / 155 / 258 |
| W53 | 128 px | 39.5 | 48 / 55 / 60 | 44 / 59 / 74 | 96 / 110 / 120 | 48 / 82 / 143 | 124 / 216 / 366 |
| W53 | 256 px | 79.0 | 82 / 103 / 115 | 84 / 112 / 145 | 164 / 206 / 230 | 81 / 93 / 145 | 167 / 305 / 524 |

**Summary over all 96 cases** (32 configurations × 3 values of N). Ratios are rule ÷
measured minimum, so ≥ 1 means the rule is safe.

| subset | min n_coarse / K | 2K rule: min | median | max | n_exact rule: min | median | max |
|---|---|---|---|---|---|---|---|
| all | 0.75–1.50 | 1.33 | 1.92 | 2.67 | 1.31 | 2.26 | 4.74 |
| N = 1e8 | 0.75–1.17 | 1.71 | 2.00 | 2.67 | 1.35 | 2.24 | 3.22 |
| N = 1e9 | 0.77–1.25 | 1.60 | 1.88 | 2.58 | 1.36 | 2.28 | 3.52 |
| N = 1e10 | 1.00–1.50 | 1.33 | 1.60 | 2.00 | 1.31 | 2.21 | 4.74 |

At the rules' own predictions, the largest Δχ² was 0.056 for Hermite (2K) and 0.32 for
exact propagation.

## Conclusion

**Neither rule undershot** in any of the 96 cases (Δχ² < 1 at the prediction in 96/96 for
both). The generic setup reproduces the NIRCam study closely: K is typically 5–20% above the
NIRCam values for the matching filter and stamp (by 0–1 for the narrowest filter), and
min n_coarse / K spans 0.75–1.50 (NIRCam: 0.87–1.78). Both rules are confirmed, and both
are conservative:

- **Hermite, `max(3, 2K)`:** at least 1.33× the minimum (median 1.9×). The tightest cases
  are at N = 1e10. `max(3, ceil(1.45 K))` would already have sufficed here (NIRCam needed
  1.70), so `c = 2` leaves about 20–40% headroom for setups less benign than these.
- **Exact, `23.5 K^0.553 (N/1e9)^0.208`:** at least 1.31× the minimum, but with a median of
  2.3× and up to 4.7×. The overshoot grows with stamp size. At fixed filter and N, the
  minimum `n_exact` barely depends on the stamp (W29 at 1e10: 91–134 from 32 to 256 px),
  while K roughly triples, so the `K^0.553` term over-predicts for large stamps.

Caveats:

- **The speedup estimate is biased towards Hermite.** `recommend_wavelength_sampling`
  estimates the speedup as `n_exact / (2.5 n_coarse)` with the conservative `n_exact`
  rule. Against the measured minimum `n_exact`, it predicts ≥ 1.5× when the true figure is
  below 1.5× in 36 of 96 cases, mostly narrow filters or small K at high N (e.g. N3 at
  32 px: estimated 1.8–4.1×, true 0.6–1.2×). The recommendation is still safe (both
  rules are), but it can pick the slower option. The `jvp_cost` of 2.5
  was not re-measured here.
- **The filters are smooth.** The quadrature error, and so `n_exact`, depends on how sharp
  the throughput is. Real filters with steeper edges or ripples need more bins. At N = 1e10,
  NIRCam's F322W2 needed 150–200 (on a coarse grid), against 103–145 for W53 here. Check
  `n_exact` with a real throughput curve before relying on a margin as small as 1.3×.
- **The tested range is narrow.** These results cover one source position, a flat source
  spectrum, Poisson noise only, 600 fine bins, and N = 1e8–1e10, the range the rules were
  fitted on.
