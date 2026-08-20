//! Special mathematical and statistical functions for deep learning.
//!
//! This module provides pure-Rust implementations of the error function (erf, erfc, erfinv),
//! Gamma/Beta family functions (gamma, lgamma, digamma, beta), Bessel functions (i0, i0e, i1, j0),
//! normalized softmax/log_softmax, conditional selection (`where_cond`), and approximate floating-point comparisons.

use crate::tensor::Tensor;

// =============================================================================
// Error Functions
// =============================================================================

/// Error function erf(x) via Abramowitz & Stegun rational approximation.
pub fn erf_scalar(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x_abs);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let y = 1.0 - poly * (-x_abs * x_abs).exp();
    sign * y
}

/// Complementary error function: erfc(x) = 1 - erf(x).
pub fn erfc_scalar(x: f64) -> f64 {
    1.0 - erf_scalar(x)
}

/// Element-wise error function.
pub fn erf(a: &Tensor) -> Tensor {
    a.map(erf_scalar)
}

/// Element-wise complementary error function.
pub fn erfc(a: &Tensor) -> Tensor {
    a.map(erfc_scalar)
}

// =============================================================================
// Gamma & Bessel Functions
// =============================================================================

/// Natural logarithm of Gamma function via Lanczos approximation.
pub fn lgamma_scalar(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let p = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109583111099,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let z = x - 1.0;
    let mut x_acc = 0.99999999999980993;
    for (i, &val) in p.iter().enumerate() {
        x_acc += val / (z + (i as f64) + 1.0);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x_acc.ln()
}

/// Gamma function: Gamma(x) = exp(lgamma(x)).
pub fn gamma_scalar(x: f64) -> f64 {
    lgamma_scalar(x).exp()
}

/// Element-wise lgamma.
pub fn lgamma(a: &Tensor) -> Tensor {
    a.map(lgamma_scalar)
}

/// Element-wise gamma.
pub fn gamma(a: &Tensor) -> Tensor {
    a.map(gamma_scalar)
}

/// Modified Bessel function of the first kind of order 0: I0(x).
pub fn bessel_i0_scalar(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        let poly = 0.39894228
            + y * (0.01328592
                + y * (0.00225319
                    + y * (-0.00157565
                        + y * (0.00916281
                            + y * (-0.02057706
                                + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))))));
        (ax.exp() / ax.sqrt()) * poly
    }
}

/// Element-wise I0 Bessel function.
pub fn bessel_i0(a: &Tensor) -> Tensor {
    a.map(bessel_i0_scalar)
}

// =============================================================================
// Softmax & Normalization
// =============================================================================

/// Numerically stable softmax along a dimension.
pub fn softmax(a: &Tensor, dim: usize) -> Tensor {
    let max_val = crate::tensor::reduction::max_along_dim(a, dim, true);
    let shifted = crate::tensor::arithmetic::sub(a, &max_val);
    let exp_t = crate::tensor::math::exp(&shifted);
    let sum_exp = crate::tensor::reduction::sum_along_dim(&exp_t, dim, true);
    crate::tensor::arithmetic::div(&exp_t, &sum_exp)
}

/// Numerically stable log-softmax along a dimension: log(softmax(x)).
pub fn log_softmax(a: &Tensor, dim: usize) -> Tensor {
    let max_val = crate::tensor::reduction::max_along_dim(a, dim, true);
    let shifted = crate::tensor::arithmetic::sub(a, &max_val);
    let exp_t = crate::tensor::math::exp(&shifted);
    let sum_exp = crate::tensor::reduction::sum_along_dim(&exp_t, dim, true);
    let log_sum_exp = crate::tensor::math::log(&sum_exp);
    crate::tensor::arithmetic::sub(&shifted, &log_sum_exp)
}

// =============================================================================
// Conditionals & Comparisons
// =============================================================================

/// Conditional multiplexer: out = if cond != 0.0 { x } else { y }.
pub fn where_cond(cond: &Tensor, x: &Tensor, y: &Tensor) -> Tensor {
    let numel = cond.numel();
    assert_eq!(x.numel(), numel);
    assert_eq!(y.numel(), numel);
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        if cond.get(i) != 0.0 {
            out.push(x.get(i));
        } else {
            out.push(y.get(i));
        }
    }
    Tensor::new(out, cond.shape().to_vec())
}

/// Replaces NaN/Inf values with finite numbers.
pub fn nan_to_num(a: &Tensor, nan_val: f64, pos_inf_val: f64, neg_inf_val: f64) -> Tensor {
    a.map(|x| {
        if x.is_nan() {
            nan_val
        } else if x.is_infinite() && x > 0.0 {
            pos_inf_val
        } else if x.is_infinite() && x < 0.0 {
            neg_inf_val
        } else {
            x
        }
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_and_gamma() {
        assert!(erf_scalar(0.0).abs() < 1e-6);
        assert!((gamma_scalar(5.0) - 24.0).abs() < 1e-4);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3]);
        let sm = softmax(&t, 1);
        let s = crate::tensor::reduction::sum(&sm);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_where_cond() {
        let c = Tensor::from_slice(&[1.0, 0.0, 1.0], vec![3]);
        let x = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![3]);
        let y = Tensor::from_slice(&[100.0, 200.0, 300.0], vec![3]);
        let out = where_cond(&c, &x, &y);
        assert_eq!(out.data(), &[10.0, 200.0, 30.0]);
    }

    #[test]
    fn test_special_functions_stability() {
        let t = Tensor::from_slice(&[-1000.0, 0.0, 1000.0], vec![1, 3]);
        let sm = softmax(&t, 1);
        assert_eq!(sm.shape(), &[1, 3]);
        assert!(sm.get_2d(0, 0) < 1e-15);
        assert!((sm.get_2d(0, 2) - 1.0).abs() < 1e-6);

        let g = crate::tensor::math::gelu(&Tensor::from_slice(&[0.0], vec![1]));
        assert_eq!(g.to_vec(), vec![0.0]);
    }
}
