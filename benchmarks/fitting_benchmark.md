# Fitting benchmark: which `FisherFit` method to use

Reproduce with:

```bash
python benchmarks/fitting_benchmark.py          # ~10 min
python benchmarks/fitting_benchmark.py --quick  # ~2 min, one start per distance
```

Run on 2026-10-15, Apple M5 Pro CPU, jax 0.11.2, dLux 0.15.1, optax 0.2.8.

## Recommendations

1. **Default: diagonal SGD with lr = 1**, i.e. `problem.fit()`. It's strictly first-order (one
   gradient per step, curvature computed once) and converged from every start here. lr = 1
   needed half the steps of lr = 0.5 and was stable everywhere. It slows down once parameters
   are correlated: 106–944 steps at n = 33.
2. **Better for most fits: L-BFGS**, `problem.fit(optimiser="lbfgs")`, ideally with
   `precondition="full"`. Each step costs about 2 gradients instead of 1 (because of the line
   search). Even so, a whole fit cost the same as the default near the minimum on the
   uncorrelated problem, and **3–17× fewer gradients** everywhere else (table below). It
   converged from all 24 starts.
3. **Momentum, if you want to stay strictly first-order:** `optimiser="nesterov"` (lr 1,
   Nesterov momentum 0.9). It's still one gradient per step with no curvature model. It took
   5–9× fewer steps than the default when parameters are correlated (n = 33). When they
   weren't (n = 11) it was slightly slower near the minimum (7 vs 6 and 14 vs 9 steps) and
   ~20% faster far away. From far starts it still costs ~1.5–3× more gradients than L-BFGS. Plain (heavy-ball) momentum 0.9 is worse: it overshoots on easy
   problems (39 vs 6 steps at 3σ, n = 11). **Avoid Adam:** its per-parameter rescaling undoes
   the Fisher whitening, and it never converged from 300σ or the physical start.
4. **Already near the minimum** (a re-fit, a refined model, or after a coarse fit): use
   **full SGD with lr = 1**, `problem.fit(precondition="full")`. This is a natural-gradient
   (Newton) step, and it converged in 1–3 steps from 3σ and 30σ. **Do not use it from far
   away:** from 300σ it converged on only 2/3 starts at n = 11 and 0/3 at n = 33.
5. **Refreshing** the curvature after 5 steps of full SGD (lr 1) cut the physical-start fit
   from 32 to 9 steps (n = 11) and from 54 to 11 (n = 33). From 300σ it doesn't help
   reliably (2/3 and 1/3 converged). L-BFGS gets the same benefit without a refresh.
6. **Old `spectrum` default:** diag(F⁻¹) preconditioning, normalised by λ_max. It was the
   slowest method everywhere. It took the same number of steps as the new 1/diag(F) diagonal
   SGD (at lr 0.5) on the easy problem near the minimum, and 10–35% more elsewhere.
7. **Uncertainties:** whichever method fits, compute
   `problem.curvature(fit.params, precondition="full").std` at the result.

## Cost per fit, including the slower L-BFGS steps

Gradients per fit = median steps × (time per step ÷ time per gradient), excluding the
one-off curvature setup. The default and diagonal L-BFGS share the same setup (the diagonal
Fisher), so they compare directly.

| n | method | 3σ | 30σ | 300σ | physical |
|---|---|---|---|---|---|
| 11 | diagonal SGD, lr 1 (default) | 6 | 9 | 86 | 81 |
| 11 | diagonal SGD, Nesterov 0.9 | 7 | 14 | 67 | 67 |
| 11 | diagonal L-BFGS | 6 | 10 | 28 | 22 |
| 11 | full L-BFGS | 4 | 9 | 28 | 24 |
| 11 | full SGD, lr 1 | 1 | 3 | 58 (2/3 converged) | 32 |
| 33 | diagonal SGD, lr 1 (default) | 106 | 254 | 735 | 944 |
| 33 | diagonal SGD, Nesterov 0.9 | 19 | 49 | 93 | 107 |
| 33 | diagonal L-BFGS | 24 | 42 | 58 | 56 |
| 33 | full L-BFGS | 4 | 8 | 33 | 46 |
| 33 | full SGD, lr 1 | 1 | 3 | none converged | 59 |

## Cost of the curvature

At these sizes the **full Fisher was cheaper to compute than the diagonal**: 0.11 s vs 0.15 s
at n = 11, and 0.13 s vs 0.17 s at n = 33. The diagonal's 20 power iterations, used to
estimate λ_max, cost more than the extra columns of the full matrix. Both timings include
JIT compilation, which dominates for a model this cheap.

For a slow, real model where compilation is negligible, rough costs are:
- **Diagonal Fisher:** about n forward-mode passes plus about 60 gradients for the power
  iteration.
- **Full Fisher:** about 2–3 gradients per parameter.

So the diagonal only becomes cheaper beyond a few tens to ~100 parameters. Its fits also
need far more steps, so in total cost the full Fisher stays ahead until n reaches the
hundreds. That's also roughly where the n × n matrix and its O(n³) Cholesky start to matter.

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
  (≲ 0.15σ in the worst direction).
- **Cells** show median steps (starts converged / starts) / wall time. Wall time = curvature
  setup + steps × ms per step after JIT warm-up. Caps: 3000 SGD steps, 300 L-BFGS steps.
- **One gradient** (`value_and_grad`) took about 1.1 ms. An SGD step costs one gradient; an
  L-BFGS step costs about two, because of the line search.

## Results

### n = 11 parameters (8 Zernikes + position + flux), cond(F_n) at truth = 2.95

Curvature setup (incl. compile): diagonal 0.15 s (~137 gradients), full 0.11 s (~94 gradients). One value_and_grad ~ 1.1 ms.

| method | ms/step | 3σ | 30σ | 300σ | physical |
|---|---|---|---|---|---|
| diagonal SGD, lr 1.0 (default) | 1.1 | 6 (3/3) / 0.16 s | 9 (3/3) / 0.16 s | 86 (3/3) / 0.25 s | 81 (1/1) / 0.24 s |
| diagonal SGD, lr 0.5 | 1.1 | 13 (3/3) / 0.17 s | 21 (3/3) / 0.18 s | 176 (3/3) / 0.35 s | 166 (1/1) / 0.34 s |
| diagonal SGD, Nesterov (lr 1, momentum 0.9) | 1.1 | 7 (3/3) / 0.16 s | 14 (3/3) / 0.17 s | 67 (3/3) / 0.23 s | 67 (1/1) / 0.23 s |
| diagonal SGD, heavy-ball momentum 0.9 | 1.1 | 39 (3/3) / 0.20 s | 86 (3/3) / 0.25 s | 141 (3/3) / 0.31 s | 161 (1/1) / 0.34 s |
| diagonal Adam, lr 0.1 (not recommended) | 1.1 | 45 (3/3) / 0.20 s | 417 (3/3) / 0.63 s | not converged (0/3) | not converged (0/1) |
| full SGD, lr 0.5 | 1.1 | 4 (3/3) / 0.11 s | 8 (3/3) / 0.11 s | 122 (3/3) / 0.24 s | 69 (1/1) / 0.18 s |
| full SGD, lr 1.0 | 1.1 | 1 (3/3) / 0.11 s | 3 (3/3) / 0.11 s | 58 (2/3) / 0.17 s | 32 (1/1) / 0.14 s |
| full SGD, lr 1.0, refresh at step 5 | 1.1 | 1 (3/3) / 0.22 s | 3 (3/3) / 0.22 s | 11 (2/3) / 0.23 s | 9 (1/1) / 0.23 s |
| diagonal L-BFGS | 2.2 | 3 (3/3) / 0.16 s | 5 (3/3) / 0.16 s | 14 (3/3) / 0.18 s | 11 (1/1) / 0.18 s |
| **full L-BFGS** | 2.3 | 2 (3/3) / 0.11 s | 4 (3/3) / 0.11 s | 13 (3/3) / 0.14 s | 11 (1/1) / 0.13 s |
| baseline: old diag(F⁻¹) SGD, lr 0.5 | 1.1 | 13 (3/3) / 0.27 s | 21 (3/3) / 0.28 s | 195 (3/3) / 0.48 s | 200 (1/1) / 0.49 s |

### n = 33 parameters (30 Zernikes + position + flux), cond(F_n) at truth = 94

Curvature setup (incl. compile): diagonal 0.17 s (~151 gradients), full 0.13 s (~113 gradients). One value_and_grad ~ 1.1 ms.

| method | ms/step | 3σ | 30σ | 300σ | physical |
|---|---|---|---|---|---|
| diagonal SGD, lr 1.0 (default) | 1.1 | 106 (3/3) / 0.30 s | 254 (3/3) / 0.47 s | 735 (3/3) / 1.02 s | 944 (1/1) / 1.26 s |
| diagonal SGD, lr 0.5 | 1.1 | 212 (3/3) / 0.42 s | 508 (3/3) / 0.75 s | 1472 (3/3) / 1.85 s | 1890 (1/1) / 2.33 s |
| diagonal SGD, Nesterov (lr 1, momentum 0.9) | 1.1 | 19 (3/3) / 0.20 s | 49 (3/3) / 0.23 s | 93 (3/3) / 0.28 s | 107 (1/1) / 0.30 s |
| diagonal SGD, heavy-ball momentum 0.9 | 1.1 | 49 (3/3) / 0.23 s | 93 (3/3) / 0.28 s | 139 (3/3) / 0.33 s | 160 (1/1) / 0.36 s |
| diagonal Adam, lr 0.1 (not recommended) | 1.2 | 66 (3/3) / 0.25 s | 1033 (3/3) / 1.38 s | not converged (0/3) | not converged (0/1) |
| full SGD, lr 0.5 | 1.2 | 5 (3/3) / 0.14 s | 9 (3/3) / 0.14 s | 88 (2/3) / 0.23 s | 106 (1/1) / 0.25 s |
| full SGD, lr 1.0 | 1.2 | 1 (3/3) / 0.13 s | 3 (3/3) / 0.13 s | not converged (0/3) | 54 (1/1) / 0.19 s |
| full SGD, lr 1.0, refresh at step 5 | 1.2 | 1 (3/3) / 0.26 s | 3 (3/3) / 0.27 s | 13 (1/3) / 0.28 s | 11 (1/1) / 0.28 s |
| diagonal L-BFGS | 2.2 | 12 (3/3) / 0.20 s | 21 (3/3) / 0.22 s | 29 (3/3) / 0.24 s | 28 (1/1) / 0.24 s |
| **full L-BFGS** | 2.4 | 2 (3/3) / 0.13 s | 4 (3/3) / 0.14 s | 16 (3/3) / 0.17 s | 22 (1/1) / 0.18 s |
| baseline: old diag(F⁻¹) SGD, lr 0.5 | 1.2 | 289 (3/3) / 0.64 s | 609 (3/3) / 1.01 s | 1690 (3/3) / 2.27 s | 2371 (1/1) / 3.06 s |

## Caveats

- **The model is a cheap toy** (1 ms per gradient), so the absolute times are dominated by
  JIT compilation. Compare methods by step counts, and by setup cost in gradients, when
  extrapolating to your own models.
- **Real data is harder.** Noise, model mismatch and multiple minima (e.g. the sign ambiguity
  of the even Zernikes) make things harder than this noiseless single-source case.
- **Low signal changes the picture.** At very low signal the Fisher can change several-fold
  within a few σ. In one 187-photon test, both diagonal and full SGD failed while L-BFGS
  converged from a far start; its near-start run was never completed. That is another reason
  to prefer L-BFGS's line search when unsure.
- **Small scale.** Only 3 random starts per distance and n ≤ 33. Rerun the script after
  changes, or adapt `make_problem` to resemble your own data.
- **Local minima are not tested.** All starts here sit in the truth's basin. None of these
  methods reliably escapes a wrong basin, momentum included: there's no gradient noise in
  full-image fits. Use multiple starts, coarse-to-fine fitting, or explicit enumeration of
  known ambiguities instead.
