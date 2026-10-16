# Unit-free Hessians

A Hessian in physical units has entries ∂²L/∂xᵢ∂xⱼ, so each row and column carries its
parameter's units. Put a Zernike coefficient in metres (with a nanometre-scale effect) next to a
flux of 1e6, and the diagonal spans ~30 decades. The matrix's condition number then reflects
your choice of units, not anything physical.

`dluxtras.curvature` holds the Hessian in **unit-free form**:

$$
H = \frac{N}{s\,s^\top}, \qquad s_i = \frac{1}{\sqrt{|H_{ii}|}}
$$

`N` (`normalised`) is dimensionless with a unit diagonal, and `s` carries the units. For a
Fisher matrix, `s_i` is the conditional 1σ uncertainty of parameter `i`, with all others fixed.
This is diagonal (Jacobi) scaling, also called equilibration. It's what LAPACK's `xPOEQU`
does, and the same step as going from a covariance to a correlation matrix.

```python
from dluxtras.curvature import scaled_hessian, scaled_covariance, correlation

normalised, s, unflatten = scaled_hessian(loss, params)   # params: a pytree, e.g. a dict
cov = scaled_covariance(normalised, s)                     # Cholesky on normalised, then rescale
sigma = unflatten(np.sqrt(np.diag(cov)))                   # back in your parameter structure
corr = correlation(normalised)                             # the scales cancel
```

For fitting, `FitProblem.curvature` computes the Fisher matrix directly in this form; see
[Fitting](fitting.md).

## What units do and don't affect

Measured on a dLux model: 8 Zernike coefficients in metres, position in radians, flux 1e8.

| | |
|---|---|
| diag(H) range | 1.9e-12 to 2.2e18 |
| cond(H), raw | 5.3e18 |
| cond(normalised) | **2.0** |

Comparing float32 against float64:

| float32 computation | error |
|---|---|
| Hessian entries, raw units | 1.3e-5 |
| Hessian entries, rescaled before differentiating | 1.3e-5 (identical) |
| Marginal σ via `inv(H)` (LU) on the raw matrix | 6e-6 |
| Marginal σ via `pinv(H)` or `eigh(H)` on the raw matrix | **100% wrong**, in float64 too |
| Marginal σ via `scaled_covariance` | 6e-6 |

- **The Hessian itself is not unit-sensitive.** Floating point has relative precision, and
  autodiff applies the chain rule exactly. Rescaling a parameter just rescales its row and
  column, so rescaling before differentiating gains nothing, except protection from float
  over/underflow (float32 covers roughly 1e±38).
- **Linear algebra that adds to or thresholds the matrix is unit-sensitive.** That covers
  `pinv`, `eigh`, SVD and `rcond` cutoffs (all relative to the largest entry), damping or ridge
  terms (`+ μI`), and step-size normalisation. In raw units these silently discard the
  small-scale parameters; here they did so even in float64, since 5e18 exceeds float64's
  ~1e16 range of precision. Do them on `normalised`.
- **Real degeneracy survives normalisation.** Correlations near ±1 between parameters are
  physical. `cond(normalised)` measures them, and no diagonal rescaling can do much better
  (within a factor of n; van der Sluis, 1969). In float32 you lose about
  `log10(cond(normalised))` of the ~7 available digits, on top of the model's own float32
  precision.

## Float32 or float64?

Normalisation removes the *units* reason to need float64. Three reasons can remain:

1. **Degeneracy.** The error in the uncertainties is roughly (relative precision of the
   Hessian entries) × cond(normalised). With float32 models giving ~1e-5 entries, cond 1e3
   already means ~1% errors.
2. **Large values with small uncertainties.** If σ/|x| < ~1e-7, float32 cannot represent the
   parameter's value precisely enough, and no rescaling fixes that.
3. **Loss differences.** A float32 negative log likelihood summed over a large image cannot
   resolve changes below ~0.1, while line searches and convergence tests need much smaller
   differences. This is why `dluxtras.fitting` requires `jax_enable_x64`.
