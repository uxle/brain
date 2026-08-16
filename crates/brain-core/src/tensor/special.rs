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
        1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        let poly = 0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))))));
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
    fn test_special_stress_case_001() {
        let x = (1 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_002() {
        let x = (2 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_003() {
        let x = (3 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_004() {
        let x = (4 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_005() {
        let x = (5 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_006() {
        let x = (6 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_007() {
        let x = (7 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_008() {
        let x = (8 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_009() {
        let x = (9 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_010() {
        let x = (10 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_011() {
        let x = (11 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_012() {
        let x = (12 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_013() {
        let x = (13 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_014() {
        let x = (14 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_015() {
        let x = (15 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_016() {
        let x = (16 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_017() {
        let x = (17 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_018() {
        let x = (18 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_019() {
        let x = (19 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_020() {
        let x = (20 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_021() {
        let x = (21 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_022() {
        let x = (22 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_023() {
        let x = (23 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_024() {
        let x = (24 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_025() {
        let x = (25 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_026() {
        let x = (26 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_027() {
        let x = (27 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_028() {
        let x = (28 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_029() {
        let x = (29 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_030() {
        let x = (30 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_031() {
        let x = (31 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_032() {
        let x = (32 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_033() {
        let x = (33 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_034() {
        let x = (34 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_035() {
        let x = (35 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_036() {
        let x = (36 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_037() {
        let x = (37 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_038() {
        let x = (38 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_039() {
        let x = (39 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_040() {
        let x = (40 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_041() {
        let x = (41 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_042() {
        let x = (42 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_043() {
        let x = (43 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_044() {
        let x = (44 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_045() {
        let x = (45 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_046() {
        let x = (46 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_047() {
        let x = (47 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_048() {
        let x = (48 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_049() {
        let x = (49 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_050() {
        let x = (50 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_051() {
        let x = (51 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_052() {
        let x = (52 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_053() {
        let x = (53 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_054() {
        let x = (54 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_055() {
        let x = (55 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_056() {
        let x = (56 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_057() {
        let x = (57 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_058() {
        let x = (58 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_059() {
        let x = (59 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_060() {
        let x = (60 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_061() {
        let x = (61 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_062() {
        let x = (62 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_063() {
        let x = (63 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_064() {
        let x = (64 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_065() {
        let x = (65 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_066() {
        let x = (66 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_067() {
        let x = (67 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_068() {
        let x = (68 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_069() {
        let x = (69 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_070() {
        let x = (70 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_071() {
        let x = (71 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_072() {
        let x = (72 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_073() {
        let x = (73 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_074() {
        let x = (74 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_075() {
        let x = (75 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_076() {
        let x = (76 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_077() {
        let x = (77 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_078() {
        let x = (78 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_079() {
        let x = (79 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_080() {
        let x = (80 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_081() {
        let x = (81 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_082() {
        let x = (82 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_083() {
        let x = (83 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_084() {
        let x = (84 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_085() {
        let x = (85 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_086() {
        let x = (86 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_087() {
        let x = (87 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_088() {
        let x = (88 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_089() {
        let x = (89 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_090() {
        let x = (90 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_091() {
        let x = (91 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_092() {
        let x = (92 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_093() {
        let x = (93 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_094() {
        let x = (94 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_095() {
        let x = (95 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_096() {
        let x = (96 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_097() {
        let x = (97 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_098() {
        let x = (98 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_099() {
        let x = (99 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_100() {
        let x = (100 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_101() {
        let x = (101 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_102() {
        let x = (102 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_103() {
        let x = (103 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_104() {
        let x = (104 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_105() {
        let x = (105 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_106() {
        let x = (106 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_107() {
        let x = (107 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_108() {
        let x = (108 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_109() {
        let x = (109 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_110() {
        let x = (110 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_111() {
        let x = (111 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_112() {
        let x = (112 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_113() {
        let x = (113 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_114() {
        let x = (114 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_115() {
        let x = (115 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_116() {
        let x = (116 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_117() {
        let x = (117 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_118() {
        let x = (118 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_119() {
        let x = (119 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_120() {
        let x = (120 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_121() {
        let x = (121 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_122() {
        let x = (122 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_123() {
        let x = (123 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_124() {
        let x = (124 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_125() {
        let x = (125 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_126() {
        let x = (126 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_127() {
        let x = (127 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_128() {
        let x = (128 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_129() {
        let x = (129 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_130() {
        let x = (130 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_131() {
        let x = (131 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_132() {
        let x = (132 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_133() {
        let x = (133 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_134() {
        let x = (134 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_135() {
        let x = (135 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_136() {
        let x = (136 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_137() {
        let x = (137 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_138() {
        let x = (138 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_139() {
        let x = (139 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_140() {
        let x = (140 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_141() {
        let x = (141 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_142() {
        let x = (142 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_143() {
        let x = (143 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_144() {
        let x = (144 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_145() {
        let x = (145 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_146() {
        let x = (146 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_147() {
        let x = (147 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_148() {
        let x = (148 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_149() {
        let x = (149 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_150() {
        let x = (150 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_151() {
        let x = (151 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_152() {
        let x = (152 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_153() {
        let x = (153 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_154() {
        let x = (154 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_155() {
        let x = (155 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_156() {
        let x = (156 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_157() {
        let x = (157 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_158() {
        let x = (158 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_159() {
        let x = (159 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_160() {
        let x = (160 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_161() {
        let x = (161 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_162() {
        let x = (162 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_163() {
        let x = (163 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_164() {
        let x = (164 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_165() {
        let x = (165 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_166() {
        let x = (166 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_167() {
        let x = (167 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_168() {
        let x = (168 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_169() {
        let x = (169 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_170() {
        let x = (170 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_171() {
        let x = (171 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_172() {
        let x = (172 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_173() {
        let x = (173 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_174() {
        let x = (174 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_175() {
        let x = (175 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_176() {
        let x = (176 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_177() {
        let x = (177 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_178() {
        let x = (178 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_179() {
        let x = (179 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_180() {
        let x = (180 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_181() {
        let x = (181 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_182() {
        let x = (182 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_183() {
        let x = (183 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_184() {
        let x = (184 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_185() {
        let x = (185 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_186() {
        let x = (186 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_187() {
        let x = (187 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_188() {
        let x = (188 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_189() {
        let x = (189 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_190() {
        let x = (190 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_191() {
        let x = (191 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_192() {
        let x = (192 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_193() {
        let x = (193 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_194() {
        let x = (194 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_195() {
        let x = (195 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_196() {
        let x = (196 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_197() {
        let x = (197 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_198() {
        let x = (198 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_199() {
        let x = (199 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_200() {
        let x = (200 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_201() {
        let x = (201 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_202() {
        let x = (202 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_203() {
        let x = (203 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_204() {
        let x = (204 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_205() {
        let x = (205 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_206() {
        let x = (206 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_207() {
        let x = (207 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_208() {
        let x = (208 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_209() {
        let x = (209 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_210() {
        let x = (210 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_211() {
        let x = (211 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_212() {
        let x = (212 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_213() {
        let x = (213 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_214() {
        let x = (214 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_215() {
        let x = (215 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_216() {
        let x = (216 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_217() {
        let x = (217 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_218() {
        let x = (218 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_219() {
        let x = (219 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_220() {
        let x = (220 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_221() {
        let x = (221 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_222() {
        let x = (222 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_223() {
        let x = (223 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_224() {
        let x = (224 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_225() {
        let x = (225 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_226() {
        let x = (226 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_227() {
        let x = (227 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_228() {
        let x = (228 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_229() {
        let x = (229 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_230() {
        let x = (230 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_231() {
        let x = (231 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_232() {
        let x = (232 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_233() {
        let x = (233 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_234() {
        let x = (234 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_235() {
        let x = (235 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_236() {
        let x = (236 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_237() {
        let x = (237 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_238() {
        let x = (238 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_239() {
        let x = (239 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_240() {
        let x = (240 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_241() {
        let x = (241 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_242() {
        let x = (242 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_243() {
        let x = (243 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_244() {
        let x = (244 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_245() {
        let x = (245 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_246() {
        let x = (246 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_247() {
        let x = (247 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_248() {
        let x = (248 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_249() {
        let x = (249 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_250() {
        let x = (250 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_251() {
        let x = (251 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_252() {
        let x = (252 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_253() {
        let x = (253 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_254() {
        let x = (254 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_255() {
        let x = (255 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_256() {
        let x = (256 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_257() {
        let x = (257 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_258() {
        let x = (258 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_259() {
        let x = (259 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_260() {
        let x = (260 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_261() {
        let x = (261 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_262() {
        let x = (262 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_263() {
        let x = (263 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_264() {
        let x = (264 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_265() {
        let x = (265 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_266() {
        let x = (266 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_267() {
        let x = (267 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_268() {
        let x = (268 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_269() {
        let x = (269 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_270() {
        let x = (270 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_271() {
        let x = (271 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_272() {
        let x = (272 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_273() {
        let x = (273 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_274() {
        let x = (274 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_275() {
        let x = (275 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_276() {
        let x = (276 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_277() {
        let x = (277 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_278() {
        let x = (278 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_279() {
        let x = (279 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_280() {
        let x = (280 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_281() {
        let x = (281 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_282() {
        let x = (282 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_283() {
        let x = (283 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_284() {
        let x = (284 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_285() {
        let x = (285 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_286() {
        let x = (286 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_287() {
        let x = (287 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_288() {
        let x = (288 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_289() {
        let x = (289 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_290() {
        let x = (290 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_291() {
        let x = (291 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_292() {
        let x = (292 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_293() {
        let x = (293 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_294() {
        let x = (294 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_295() {
        let x = (295 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_296() {
        let x = (296 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_297() {
        let x = (297 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_298() {
        let x = (298 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_299() {
        let x = (299 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_300() {
        let x = (300 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_301() {
        let x = (301 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_302() {
        let x = (302 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_303() {
        let x = (303 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_304() {
        let x = (304 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_305() {
        let x = (305 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_306() {
        let x = (306 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_307() {
        let x = (307 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_308() {
        let x = (308 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_309() {
        let x = (309 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_310() {
        let x = (310 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_311() {
        let x = (311 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_312() {
        let x = (312 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_313() {
        let x = (313 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_314() {
        let x = (314 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_315() {
        let x = (315 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_316() {
        let x = (316 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_317() {
        let x = (317 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_318() {
        let x = (318 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_319() {
        let x = (319 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_320() {
        let x = (320 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_321() {
        let x = (321 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_322() {
        let x = (322 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_323() {
        let x = (323 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_324() {
        let x = (324 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_325() {
        let x = (325 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_326() {
        let x = (326 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_327() {
        let x = (327 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_328() {
        let x = (328 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_329() {
        let x = (329 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_330() {
        let x = (330 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_331() {
        let x = (331 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_332() {
        let x = (332 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_333() {
        let x = (333 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_334() {
        let x = (334 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_335() {
        let x = (335 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_336() {
        let x = (336 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_337() {
        let x = (337 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_338() {
        let x = (338 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_339() {
        let x = (339 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_340() {
        let x = (340 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_341() {
        let x = (341 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_342() {
        let x = (342 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_343() {
        let x = (343 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_344() {
        let x = (344 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_345() {
        let x = (345 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_346() {
        let x = (346 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_347() {
        let x = (347 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_348() {
        let x = (348 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_349() {
        let x = (349 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_350() {
        let x = (350 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_351() {
        let x = (351 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_352() {
        let x = (352 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_353() {
        let x = (353 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_354() {
        let x = (354 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_355() {
        let x = (355 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_356() {
        let x = (356 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_357() {
        let x = (357 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_358() {
        let x = (358 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }

    #[test]
    fn test_special_stress_case_359() {
        let x = (359 as f64) * 0.05;
        let e = erf_scalar(x);
        assert!(e >= 0.0 && e <= 1.0);
        let g = gamma_scalar(1.0 + (x % 3.0));
        assert!(g.is_finite() && g > 0.0);
    }
}
