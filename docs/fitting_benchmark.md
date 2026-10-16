# Fitting benchmark: which `FisherFit` method to use

Reproduce with:

```bash
python benchmarks/fitting_benchmark.py          # ~10 min
python benchmarks/fitting_benchmark.py --quick  # ~1 min, one start per distance
python benchmarks/loop_speed.py                 # compiled loops and optimistix (optional)
python benchmarks/optimistix_bfgs.py           # optimistix BFGS inside FisherFit (optional)
```

Run on 2026-10-16, Apple M5 Pro CPU, jax 0.11.2, dLux 0.15.1, optax 0.2.8, optimistix 0.1.0.

## The three main options

| | Call | Per step | Use when |
|---|---|---|---|
| 1 | `problem.fit()` | 1 gradient | Default. Diagonal Fisher whitening + SGD with Nesterov momentum (lr 1, 0.9). Strictly first-order, curvature computed once. |
| 2 | `problem.fit(optimiser="lbfgs")`, or with `precondition="full"` | ~1 gradient | Fewest gradients per fit almost everywhere, converged from every start, and stops on its own. Use `"full"` when n is modest (up to a few hundred). |
| 3 | `problem.fit(precondition="full")` | 1 gradient | Natural gradient (plain SGD, lr 1). Already near the minimum: 1–3 steps. Unreliable from far away. |

Whichever you use, get uncertainties from `problem.curvature(fit.params, precondition="full").std`.

## Recommendations

1. **The default (Nesterov, diagonal)** converged from every start. It took 7–67 steps for the
   uncorrelated problem (n = 11) and 19–107 for the correlated one (n = 33), versus 6–86 and
   106–944 for plain SGD at lr 1. Nesterov at lr 1 is only stable while the whitened curvature
   stays below ~1.36 (plain SGD: 2). It never failed here, but if a fit oscillates, use
   `optax.sgd(0.7, 0.9, nesterov=True)`, which costs ~10–20% more steps.
2. **L-BFGS** costs about one gradient per step: its line search usually accepts the first
   trial point, and the next step reuses that value and gradient. It needed the fewest
   gradients per fit in almost every case (table below), and converged from all 24 starts
   with either preconditioner. It stops on its own once converged
   ([below](#stopping-on-convergence)); avoid forcing it far past convergence (`tol=0`),
   since at the exact minimum its line search struggles and steps cost ~10× more.
3. **Natural gradient** (full whitening, plain SGD lr 1) converged in 1–3 steps from 3σ and
   30σ. From 300σ it converged on only 2/3 starts (n = 11) and 0/3 (n = 33).
   Refreshing the curvature after 5 steps helped from the physical start (32 → 9 and
   54 → 11 steps) but not reliably from 300σ.
4. **Don't add momentum to full whitening.** The whitened curvature is already ~identity, so
   momentum has nothing to accelerate and only overshoots: Nesterov with `"full"` failed from
   the physical start at both sizes. That's why the default optimiser depends on the
   preconditioner.
5. **Avoid Adam.** Its per-parameter rescaling undoes the Fisher whitening; it never
   converged from 300σ or the physical start. Plain (heavy-ball) momentum 0.9 overshoots on
   easy problems (39 steps vs Nesterov's 7 at 3σ, n = 11).
6. **Old `spectrum` default** (diag(F⁻¹) preconditioning normalised by λ_max, lr 0.5) was the
   slowest method everywhere: 13–2371 steps.

## Cost per fit

Gradients per fit = median steps × gradients per step: 1 for SGD variants, ~1.1 for L-BFGS
(1.2–1.3 ms per step against 1.1–1.2 ms per gradient). The one-off curvature setup is excluded.

| n | method | 3σ | 30σ | 300σ | physical |
|---|---|---|---|---|---|
| 11 | 1. diagonal Nesterov (default) | 7 | 14 | 67 | 67 |
| 11 | 2. diagonal L-BFGS | 3 | 6 | 15 | 12 |
| 11 | 2. full L-BFGS | 2 | 4 | 14 | 12 |
| 11 | 3. natural gradient | 1 | 3 | 58 (2/3 converged) | 32 |
| 11 | plain SGD, lr 1 | 6 | 9 | 86 | 81 |
| 33 | 1. diagonal Nesterov (default) | 19 | 49 | 93 | 107 |
| 33 | 2. diagonal L-BFGS | 13 | 23 | 32 | 31 |
| 33 | 2. full L-BFGS | 2 | 4 | 17 | 24 |
| 33 | 3. natural gradient | 1 | 3 | none converged | 54 |
| 33 | plain SGD, lr 1 | 106 | 254 | 735 | 944 |

## Stopping on convergence

`FisherFit.run` stops once two consecutive steps each move every parameter by less than `tol`
conditional sigma; L-BFGS uses `tol = 0.01` by default. For each tolerance below, the cells
show steps until the stop / steps until the loss was within 0.01 of the minimum / final error
in marginal sigma. Each is the worst case over the starts at that distance.

| n | method | tol | 3σ | 30σ | 300σ | physical |
|---|---|---|---|---|---|---|
| 11 | diagonal L-BFGS | 0.1 | 5 / 4 / 0.007 | 8 / 6 / 0.003 | 18 / 14 / 0.005 | 13 / 11 / 0.004 |
| 11 | diagonal L-BFGS | **0.01** | 7 / 4 / 0.001 | 9 / 6 / 0.0002 | 19 / 14 / 0.0007 | 15 / 11 / 7e-5 |
| 11 | full L-BFGS | 0.1 | 4 / 2 / 2e-5 | 7 / 5 / 4e-5 | 15 / 13 / 0.006 | 13 / 11 / 0.006 |
| 11 | full L-BFGS | **0.01** | 5 / 2 / 6e-7 | 7 / 5 / 4e-5 | 17 / 13 / 0.0003 | 15 / 11 / 0.001 |
| 33 | diagonal L-BFGS | 0.1 | 19 / 15 / **0.3** | 22 / 21 / **0.2** | 31 / 29 / 0.09 | 31 / 28 / 0.01 |
| 33 | diagonal L-BFGS | **0.01** | 21 / 15 / 0.01 | 28 / 21 / 0.005 | 36 / 29 / 0.009 | 33 / 28 / 0.009 |
| 33 | full L-BFGS | 0.1 | 4 / 2 / 1e-5 | 6 / 4 / 0.0008 | 23 / 21 / 0.03 | 24 / 22 / 0.007 |
| 33 | full L-BFGS | **0.01** | 5 / 2 / 6e-7 | 7 / 4 / 5e-5 | 27 / 21 / 0.002 | 28 / 22 / 0.0007 |

With `tol = 0.01` every run stopped within 0.01σ of the minimum, 2–7 steps after the loss had
converged. Looser tolerances save a few steps but can stop early: with diagonal whitening on
the correlated problem, `tol = 0.1` stopped up to 0.3σ short, because L-BFGS had not yet
learned the correlated directions and was still taking small steps along them.

## Cost of the curvature

At these sizes the **full Fisher was cheaper to compute than the diagonal**: 0.11 s vs 0.15 s
at n = 11, and 0.13 s vs 0.18 s at n = 33. The diagonal's 20 power iterations, used to
estimate λ_max, cost more than the extra columns of the full matrix. Both timings include
JIT compilation, which dominates for a model this cheap.

For a slow, real model where compilation is negligible, rough costs are:
- **Diagonal Fisher:** about n forward-mode passes plus about 60 gradients for the power
  iteration.
- **Full Fisher:** about 2–3 gradients per parameter.

So the diagonal only becomes cheaper beyond a few tens to ~100 parameters, and fits that use
it need more steps. In total cost the full Fisher stays ahead until n reaches the hundreds,
which is also roughly where the n × n matrix and its O(n³) Cholesky start to matter.

## Setup

- **Model:** a 64-pixel pupil with Zernike aberrations (in metres), plus source position
  (radians) and flux. Point source, 3 wavelengths, ~1e6 detected photons, Gaussian noise with
  variance data + 10. Parameter units span about 15 decades.
- **Data:** noiseless, so the truth is the exact minimum.
- **Problem sizes:** n = 11 (8 Zernikes, cond(F_n) ≈ 3) and n = 33 (30 Zernikes, strongly
  correlated, cond(F_n) ≈ 94).
- **Starts:** k marginal σ from the truth along random correlated directions (3 draws each,
  loss excess ≈ k²/2). Plus one "physical" start: 10 nm rms coefficient errors, a 40 mas
  position offset and a 30% flux error.
- **Converged** means the loss is within 0.01 of the minimum, statistically negligible
  (≲ 0.15σ in the worst direction). Each run stops once converged.
- **Cells** show median steps (starts converged / starts) / wall time. Wall time = curvature
  setup + steps × time per step. Every step is timed along the real trajectories; the column
  shows the method's median, excluding the first step (which includes compilation).
  Caps: 3000 SGD steps, 300 L-BFGS steps.
- **One gradient** (`value_and_grad`) took about 1.1 ms. SGD and L-BFGS steps each cost
  about one gradient.

## Results

### n = 11 parameters (8 Zernikes + position + flux), cond(F_n) at truth = 2.95

Curvature setup (incl. compile): diagonal 0.17 s (~148 gradients), full 0.11 s (~99 gradients). One value_and_grad ~ 1.2 ms.

| method | ms/step | 3σ | 30σ | 300σ | physical |
|---|---|---|---|---|---|
| **1. diagonal, Nesterov lr 1 (default)** | 1.2 | 7 (3/3) / 0.18 s | 14 (3/3) / 0.19 s | 67 (3/3) / 0.25 s | 67 (1/1) / 0.25 s |
| diagonal, Nesterov lr 0.7 | 1.2 | 8 (3/3) / 0.18 s | 18 (3/3) / 0.19 s | 87 (3/3) / 0.27 s | 81 (1/1) / 0.26 s |
| diagonal, plain SGD lr 1 | 1.2 | 6 (3/3) / 0.18 s | 9 (3/3) / 0.18 s | 86 (3/3) / 0.27 s | 81 (1/1) / 0.27 s |
| diagonal, heavy-ball momentum 0.9 | 1.1 | 39 (3/3) / 0.22 s | 86 (3/3) / 0.27 s | 141 (3/3) / 0.33 s | 161 (1/1) / 0.35 s |
| diagonal, Adam lr 0.1 (not recommended) | 1.1 | 45 (3/3) / 0.22 s | 417 (3/3) / 0.65 s | not converged (0/3) | not converged (0/1) |
| **2. diagonal L-BFGS** | 1.3 | 3 (3/3) / 0.17 s | 5 (3/3) / 0.18 s | 14 (3/3) / 0.19 s | 11 (1/1) / 0.18 s |
| **2. full L-BFGS** | 1.3 | 2 (3/3) / 0.12 s | 4 (3/3) / 0.12 s | 13 (3/3) / 0.13 s | 11 (1/1) / 0.13 s |
| **3. full, natural gradient (SGD lr 1)** | 1.2 | 1 (3/3) / 0.12 s | 3 (3/3) / 0.12 s | 58 (2/3) / 0.18 s | 32 (1/1) / 0.15 s |
| full, natural gradient, refresh at step 5 | 1.2 | 1 (3/3) / 0.12 s | 3 (3/3) / 0.12 s | 11 (2/3) / 0.27 s | 9 (1/1) / 0.24 s |
| full, natural gradient SGD lr 0.5 | 1.2 | 4 (3/3) / 0.12 s | 8 (3/3) / 0.12 s | 122 (3/3) / 0.26 s | 69 (1/1) / 0.19 s |
| full, Nesterov lr 1 (not recommended) | 1.2 | 2 (3/3) / 0.12 s | 18 (3/3) / 0.14 s | 49 (1/3) / 0.17 s | not converged (0/1) |
| baseline: old diag(F⁻¹) SGD, lr 0.5 | 1.2 | 13 (3/3) / 0.30 s | 21 (3/3) / 0.31 s | 195 (3/3) / 0.51 s | 200 (1/1) / 0.52 s |

### n = 33 parameters (30 Zernikes + position + flux), cond(F_n) at truth = 94

Curvature setup (incl. compile): diagonal 0.18 s (~158 gradients), full 0.13 s (~115 gradients). One value_and_grad ~ 1.1 ms.

| method | ms/step | 3σ | 30σ | 300σ | physical |
|---|---|---|---|---|---|
| **1. diagonal, Nesterov lr 1 (default)** | 1.1 | 19 (3/3) / 0.20 s | 49 (3/3) / 0.23 s | 93 (3/3) / 0.28 s | 107 (1/1) / 0.30 s |
| diagonal, Nesterov lr 0.7 | 1.1 | 23 (3/3) / 0.20 s | 58 (3/3) / 0.24 s | 99 (3/3) / 0.29 s | 100 (1/1) / 0.29 s |
| diagonal, plain SGD lr 1 | 1.1 | 106 (3/3) / 0.29 s | 254 (3/3) / 0.46 s | 735 (3/3) / 0.99 s | 944 (1/1) / 1.22 s |
| diagonal, heavy-ball momentum 0.9 | 1.1 | 49 (3/3) / 0.23 s | 93 (3/3) / 0.28 s | 139 (3/3) / 0.33 s | 160 (1/1) / 0.36 s |
| diagonal, Adam lr 0.1 (not recommended) | 1.1 | 66 (3/3) / 0.25 s | 1033 (3/3) / 1.33 s | not converged (0/3) | not converged (0/1) |
| **2. diagonal L-BFGS** | 1.2 | 12 (3/3) / 0.19 s | 21 (3/3) / 0.20 s | 29 (3/3) / 0.21 s | 28 (1/1) / 0.21 s |
| **2. full L-BFGS** | 1.3 | 2 (3/3) / 0.13 s | 4 (3/3) / 0.13 s | 16 (3/3) / 0.15 s | 22 (1/1) / 0.16 s |
| **3. full, natural gradient (SGD lr 1)** | 1.2 | 1 (3/3) / 0.13 s | 3 (3/3) / 0.13 s | not converged (0/3) | 54 (1/1) / 0.19 s |
| full, natural gradient, refresh at step 5 | 1.2 | 1 (3/3) / 0.13 s | 3 (3/3) / 0.13 s | 13 (1/3) / 0.28 s | 11 (1/1) / 0.28 s |
| full, natural gradient SGD lr 0.5 | 1.2 | 5 (3/3) / 0.13 s | 9 (3/3) / 0.14 s | 88 (2/3) / 0.23 s | 106 (1/1) / 0.25 s |
| full, Nesterov lr 1 (not recommended) | 1.2 | 2 (3/3) / 0.13 s | 44 (3/3) / 0.18 s | not converged (0/3) | not converged (0/1) |
| baseline: old diag(F⁻¹) SGD, lr 0.5 | 1.1 | 289 (3/3) / 0.63 s | 609 (3/3) / 0.98 s | 1690 (3/3) / 2.18 s | 2371 (1/1) / 2.94 s |

## Compiled loops and optimistix

Does compiling the whole solve into one JAX loop, as `optimistix.minimise` does, make fits
faster? `benchmarks/loop_speed.py` fits the n = 33 problem from the physical start, on the
cheap model and on a ~13× heavier one. All rows except the last minimise the same
Fisher-whitened loss; the last hands optimistix the raw parameters.

### Cheap model: 64-px pupil, 3 wavelengths: n = 33, one value_and_grad = 1.2 ms

| method | steps | time | final error |
|---|---|---|---|
| FisherFit L-BFGS (stops at tol 0.01) | 28 | 34 ms | 0.0007 sigma |
| bare jitted step, Python loop | 22 | 34 ms | 0.03 sigma |
| same step, whole loop compiled | 22 | 31 ms | 0.03 sigma |
| optimistix BFGS, whitened, rtol=atol=0.0001 | 22 | 19 ms | 0.001 sigma |
| optimistix BFGS, whitened, rtol=atol=1e-08 | 38 | 33 ms | 4e-08 sigma |
| optimistix LBFGS, whitened, rtol=atol=1e-8 | 41 | 34 ms | 6e-08 sigma |
| optimistix BFGS, raw units, rtol=1e-8, atol=1e-14 | 2193 | 1268 ms | 3e+02 sigma |

### Heavy model: 256-px pupil, 10 wavelengths: n = 33, one value_and_grad = 14.6 ms

| method | steps | time | final error |
|---|---|---|---|
| FisherFit L-BFGS (stops at tol 0.01) | 22 | 328 ms | 0.001 sigma |
| bare jitted step, Python loop | 17 | 306 ms | 0.03 sigma |
| same step, whole loop compiled | 17 | 301 ms | 0.03 sigma |
| optimistix BFGS, whitened, rtol=atol=0.0001 | 22 | 281 ms | 0.0007 sigma |
| optimistix BFGS, whitened, rtol=atol=1e-08 | 36 | 425 ms | 6e-08 sigma |
| optimistix LBFGS, whitened, rtol=atol=1e-8 | 35 | 425 ms | 9e-08 sigma |
| optimistix BFGS, raw units, rtol=1e-8, atol=1e-14 | 2183 | 15956 ms | 3e+02 sigma |

- **Compiling the loop saves 2–9%.** That's the same L-BFGS steps run from Python vs inside
  one `lax.fori_loop`, i.e. the Python dispatch overhead per step. The model evaluation
  dominates, increasingly so for heavier models.
- **optimistix's BFGS is competitive in whitened coordinates**: ~15–45% faster at a loose
  tolerance (`rtol=atol=1e-4`, comparable final error), similar at a tight one. Its
  backtracking line search and dense BFGS update are as efficient as optax's L-BFGS here.
- **On raw parameters it failed:** 2000+ steps, still 300σ from the minimum. BFGS starts
  from an identity curvature, which in raw units treats a 1e-9 m coefficient and a 1e6 flux
  as the same size. The whitening matters, not the library. To use optimistix with
  whitening, minimise `lambda u: problem.loss(c.active + c.transform @ u)` with
  `c = problem.curvature(...)`.

### optimistix BFGS as FisherFit's L-BFGS?

Could optimistix's BFGS replace optax's L-BFGS *inside* `FisherFit`, which steps once per
call (for history, progress bars and resumable `run(n)`)? `benchmarks/optimistix_bfgs.py`
drives `optimistix.BFGS` that way, with the same convergence rule. Cells show steps to the
stop / steps until the loss was within 0.01 of the minimum / final error in marginal sigma,
the worst case over the starts at each distance. These timings were taken while other jobs
ran, so compare the two methods with each other, not with the tables above.

#### n = 11

| whitening | method | ms/step | 3 sigma | 30 sigma | 300 sigma | physical |
|---|---|---|---|---|---|---|
| diagonal | optax L-BFGS (FisherFit) | 1.73 | 7 / 4 / 0.001 | 9 / 6 / 0.0002 | 19 / 14 / 0.0007 | 15 / 11 / 7e-05 |
| diagonal | optimistix BFGS | 1.62 | 8 / 4 / 0.001 | 9 / 6 / 0.001 | 29 / 21 / 0.003 | 28 / 21 / 0.005 |
| full | optax L-BFGS (FisherFit) | 1.87 | 5 / 2 / 6e-07 | 7 / 5 / 4e-05 | 17 / 13 / 0.0003 | 15 / 11 / 0.001 |
| full | optimistix BFGS | 1.79 | 4 / 1 / 8e-08 | 6 / 3 / 1e-05 | 19 / 13 / 0.0009 | 16 / 12 / 0.001 |

#### n = 33

| whitening | method | ms/step | 3 sigma | 30 sigma | 300 sigma | physical |
|---|---|---|---|---|---|---|
| diagonal | optax L-BFGS (FisherFit) | 1.75 | 21 / 15 / 0.01 | 28 / 21 / 0.005 | 36 / 29 / 0.009 | 33 / 28 / 0.009 |
| diagonal | optimistix BFGS | 1.69 | 29 / 22 / 0.02 | 38 / 31 / 0.01 | 54 / 47 / 0.008 | 66 / 56 / 0.002 |
| full | optax L-BFGS (FisherFit) | 1.92 | 5 / 2 / 6e-07 | 7 / 4 / 5e-05 | 27 / 21 / 0.002 | 28 / 22 / 0.0007 |
| full | optimistix BFGS | 1.70 | 4 / 1 / 8e-07 | 6 / 3 / 5e-05 | 22 / 17 / 0.002 | 21 / 15 / 0.001 |

- **Full whitening:** optimistix took slightly fewer steps (21 vs 28 from the physical start
  at n = 33).
- **Diagonal whitening:** it took up to 2× more (66 vs 33).
- **Cost per step:** about the same.

Without the compiled loop, the speed advantage measured above disappears. Driving optimistix
step by step also relies on its solvers' `init`/`step` methods and on closure conversion,
and its `LBFGS` cannot be carried between calls this way. **So `FisherFit` keeps optax's
L-BFGS.** optimistix remains a good choice for a one-shot compiled solve in the whitened
coordinates.

## Caveats

- **The model is a cheap toy** (1 ms per gradient), so the absolute times are dominated by
  JIT compilation. Compare methods by step counts, and by setup cost in gradients, when
  extrapolating to your own models.
- **Real data is harder.** Noise, model mismatch and multiple minima (e.g. the sign ambiguity
  of the even Zernikes) make things harder than this noiseless single-source case.
- **Low signal changes the picture.** At very low signal the Fisher can change several-fold
  within a few σ. In one 187-photon test, both diagonal and full plain SGD failed while L-BFGS
  converged from a far start; its near-start run was never completed. That is another reason
  to prefer L-BFGS's line search when unsure.
- **Small scale.** Only 3 random starts per distance and n ≤ 33. Rerun the script after
  changes, or adapt `make_problem` to resemble your own data.
- **Local minima are not tested.** All starts here sit in the truth's basin. None of these
  methods reliably escapes a wrong basin, momentum included: there's no gradient noise in
  full-image fits. Use multiple starts, coarse-to-fine fitting, or explicit enumeration of
  known ambiguities instead.
