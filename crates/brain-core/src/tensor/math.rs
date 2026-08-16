//! Mathematical functions for tensors in the Brain deep learning framework.
//!
//! This module provides element-wise mathematical operations including:
//! - Exponential and logarithmic functions
//! - Trigonometric and hyperbolic functions
//! - Rounding and sign functions
//! - Activation functions (sigmoid, relu, gelu, etc.)
//! - Error functions and special functions
//!
//! All functions handle edge cases (NaN, Inf, negative values) appropriately.

use crate::tensor::Tensor;

// =============================================================================
// Exponential and Logarithmic Functions
// =============================================================================

/// Element-wise exponential: e^x.
pub fn exp(a: &Tensor) -> Tensor {
    a.map(|v| v.exp())
}

/// Element-wise natural logarithm: ln(x).
/// Returns NaN for x <= 0.
pub fn log(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v <= 0.0 { f64::NAN } else { v.ln() }
    })
}

/// Element-wise base-2 logarithm: log2(x).
pub fn log2(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v <= 0.0 { f64::NAN } else { v.log2() }
    })
}

/// Element-wise base-10 logarithm: log10(x).
pub fn log10(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v <= 0.0 { f64::NAN } else { v.log10() }
    })
}

/// Element-wise log(1 + x): numerically stable for small x.
pub fn log1p(a: &Tensor) -> Tensor {
    a.map(|v| v.ln_1p())
}

/// Element-wise logarithm with base b.
pub fn log_base(a: &Tensor, base: f64) -> Tensor {
    let inv_log_base = 1.0 / base.ln();
    a.map(move |v| {
        if v <= 0.0 { f64::NAN } else { v.ln() * inv_log_base }
    })
}

/// Element-wise x * log(x) with 0 at x=0.
pub fn xlogy(x: &Tensor, y: &Tensor) -> Tensor {
    x.map2(y, |xv, yv| {
        if yv == 0.0 { 0.0 } else { xv * yv.ln() }
    })
}

/// Element-wise 2^x.
pub fn exp2(a: &Tensor) -> Tensor {
    a.map(|v| v.exp2())
}

/// Element-wise 10^x.
pub fn exp10(a: &Tensor) -> Tensor {
    a.map(|v| 10.0_f64.powf(v))
}

// =============================================================================
// Trigonometric Functions
// =============================================================================

/// Element-wise sine.
pub fn sin(a: &Tensor) -> Tensor {
    a.map(|v| v.sin())
}

/// Element-wise cosine.
pub fn cos(a: &Tensor) -> Tensor {
    a.map(|v| v.cos())
}

/// Element-wise tangent.
pub fn tan(a: &Tensor) -> Tensor {
    a.map(|v| v.tan())
}

/// Element-wise arcsine (inverse sine) in radians.
/// Returns NaN for |x| > 1.
pub fn asin(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v < -1.0 || v > 1.0 { f64::NAN } else { v.asin() }
    })
}

/// Element-wise arccosine (inverse cosine) in radians.
/// Returns NaN for |x| > 1.
pub fn acos(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v < -1.0 || v > 1.0 { f64::NAN } else { v.acos() }
    })
}

/// Element-wise arctangent (inverse tangent) in radians.
pub fn atan(a: &Tensor) -> Tensor {
    a.map(|v| v.atan())
}

/// Element-wise two-argument arctangent: atan2(y, x).
pub fn atan2(y: &Tensor, x: &Tensor) -> Tensor {
    y.map2(x, |yv, xv| yv.atan2(xv))
}

/// Element-wise hypot: sqrt(x^2 + y^2).
pub fn atan2_tensors(y: &Tensor, x: &Tensor) -> Tensor {
    y.map2(x, |yv, xv| yv.atan2(xv))
}

// =============================================================================
// Hyperbolic Functions
// =============================================================================

/// Element-wise hyperbolic sine.
pub fn sinh(a: &Tensor) -> Tensor {
    a.map(|v| v.sinh())
}

/// Element-wise hyperbolic cosine.
pub fn cosh(a: &Tensor) -> Tensor {
    a.map(|v| v.cosh())
}

/// Element-wise hyperbolic tangent.
pub fn tanh(a: &Tensor) -> Tensor {
    a.map(|v| v.tanh())
}

/// Element-wise inverse hyperbolic sine.
pub fn asinh(a: &Tensor) -> Tensor {
    a.map(|v| v.asinh())
}

/// Element-wise inverse hyperbolic cosine.
/// Returns NaN for x < 1.
pub fn acosh(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v < 1.0 { f64::NAN } else { v.acosh() }
    })
}

/// Element-wise inverse hyperbolic tangent.
/// Returns NaN for |x| >= 1.
pub fn atanh(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v <= -1.0 || v >= 1.0 { f64::NAN } else { v.atanh() }
    })
}

// =============================================================================
// Power, Root, and Absolute Value
// =============================================================================

/// Element-wise square root: sqrt(x).
/// Returns NaN for x < 0.
pub fn sqrt(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v < 0.0 { f64::NAN } else { v.sqrt() }
    })
}

/// Element-wise cube root: cbrt(x).
/// Works for negative values.
pub fn cbrt(a: &Tensor) -> Tensor {
    a.map(|v| v.cbrt())
}

/// Element-wise absolute value: |x|.
pub fn abs(a: &Tensor) -> Tensor {
    a.map(|v| v.abs())
}

/// Element-wise sign function: -1 for negative, 0 for zero, 1 for positive.
pub fn sign(a: &Tensor) -> Tensor {
    a.map(|v| v.signum())
}

/// Element-wise reciprocal: 1/x.
pub fn reciprocal(a: &Tensor) -> Tensor {
    a.map(|v| 1.0 / v)
}

/// Element-wise square: x^2.
pub fn square(a: &Tensor) -> Tensor {
    a.map(|v| v * v)
}

/// Element-wise x^y.
pub fn pow(a: &Tensor, exponent: f64) -> Tensor {
    a.map(|v| v.powf(exponent))
}

// =============================================================================
// Rounding Functions
// =============================================================================

/// Element-wise ceiling: smallest integer >= x.
pub fn ceil(a: &Tensor) -> Tensor {
    a.map(|v| v.ceil())
}

/// Element-wise floor: largest integer <= x.
pub fn floor(a: &Tensor) -> Tensor {
    a.map(|v| v.floor())
}

/// Element-wise round to nearest integer.
pub fn round(a: &Tensor) -> Tensor {
    a.map(|v| v.round())
}

/// Element-wise truncation toward zero.
pub fn trunc(a: &Tensor) -> Tensor {
    a.map(|v| v.trunc())
}

/// Element-wise fractional part: x - floor(x).
pub fn fract(a: &Tensor) -> Tensor {
    a.map(|v| v.fract())
}

/// Element-wise round to given number of decimal places.
pub fn round_to(a: &Tensor, decimals: i32) -> Tensor {
    let factor = 10.0_f64.powi(decimals);
    a.map(move |v| (v * factor).round() / factor)
}

// =============================================================================
// Comparison and Classification Functions
// =============================================================================

/// Element-wise clamp to [min, max].
pub fn clamp(a: &Tensor, min: f64, max: f64) -> Tensor {
    a.map(|v| v.clamp(min, max))
}

/// Element-wise conditional selection: out[i] = value[i] where condition[i] else other[i].
pub fn where_tensor(condition: &Tensor, value: &Tensor, other: &Tensor) -> Tensor {
    condition.map2(value, |c, v| if c != 0.0 { v } else { 0.0 })
        .map2(other, |r, o| if r != 0.0 { r } else { o })
}

/// Element-wise conditional selection with scalar other.
pub fn where_scalar(condition: &Tensor, value: f64, other: f64) -> Tensor {
    condition.map(move |c| if c != 0.0 { value } else { other })
}

/// Returns true where tensor elements are NaN.
pub fn isnan(a: &Tensor) -> Tensor {
    a.map(|v| if v.is_nan() { 1.0 } else { 0.0 })
}

/// Returns true where tensor elements are infinite.
pub fn isinf(a: &Tensor) -> Tensor {
    a.map(|v| if v.is_infinite() { 1.0 } else { 0.0 })
}

/// Returns true where tensor elements are finite (not NaN and not Inf).
pub fn isfinite(a: &Tensor) -> Tensor {
    a.map(|v| if v.is_finite() { 1.0 } else { 0.0 })
}

/// Element-wise comparison: a < b.
pub fn less(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x < y { 1.0 } else { 0.0 })
}

/// Element-wise comparison: a <= b.
pub fn less_equal(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x <= y { 1.0 } else { 0.0 })
}

/// Element-wise comparison: a > b.
pub fn greater(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x > y { 1.0 } else { 0.0 })
}

/// Element-wise comparison: a >= b.
pub fn greater_equal(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x >= y { 1.0 } else { 0.0 })
}

/// Element-wise equality: a == b.
pub fn equal(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x == y { 1.0 } else { 0.0 })
}

/// Element-wise inequality: a != b.
pub fn not_equal(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x != y { 1.0 } else { 0.0 })
}

/// Element-wise logical NOT (treats nonzero as true).
pub fn logical_not(a: &Tensor) -> Tensor {
    a.map(|v| if v == 0.0 { 1.0 } else { 0.0 })
}

/// Element-wise logical AND (treats nonzero as true).
pub fn logical_and(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x != 0.0 && y != 0.0 { 1.0 } else { 0.0 })
}

/// Element-wise logical OR (treats nonzero as true).
pub fn logical_or(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| if x != 0.0 || y != 0.0 { 1.0 } else { 0.0 })
}

/// Element-wise logical XOR (treats nonzero as true).
pub fn logical_xor(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| {
        let xb = x != 0.0;
        let yb = y != 0.0;
        if xb ^ yb { 1.0 } else { 0.0 }
    })
}

/// Element-wise maximum (returns a where a >= b else b).
pub fn maximum(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x.max(y))
}

/// Element-wise minimum (returns a where a <= b else b).
pub fn minimum(a: &Tensor, b: &Tensor) -> Tensor {
    a.map2(b, |x, y| x.min(y))
}

// =============================================================================
// Activation Functions
// =============================================================================

/// Sigmoid activation: 1 / (1 + exp(-x)).
/// Numerically stable implementation.
pub fn sigmoid(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v >= 0.0 {
            let e = (-v).exp();
            1.0 / (1.0 + e)
        } else {
            let e = v.exp();
            e / (1.0 + e)
        }
    })
}

/// SiLU / Swish activation: x * sigmoid(x).
pub fn silu(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v >= 0.0 {
            let e = (-v).exp();
            v / (1.0 + e)
        } else {
            let e = v.exp();
            v * e / (1.0 + e)
        }
    })
}

/// Swish activation: x * sigmoid(beta * x).
pub fn swish(a: &Tensor, beta: f64) -> Tensor {
    a.map(move |v| {
        let sig = if beta * v >= 0.0 {
            let e = (-beta * v).exp();
            1.0 / (1.0 + e)
        } else {
            let e = (beta * v).exp();
            e / (1.0 + e)
        };
        v * sig
    })
}

/// Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)).
pub fn mish(a: &Tensor) -> Tensor {
    a.map(|v| {
        let softplus = if v > 20.0 { v } else if v < -20.0 { 0.0 } else { (1.0 + v.exp()).ln() };
        v * softplus.tanh()
    })
}

/// Softplus activation: ln(1 + exp(x)).
/// Numerically stable for large positive and negative values.
pub fn softplus(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v > 20.0 { v } else if v < -20.0 { 0.0 } else { (1.0 + v.exp()).ln() }
    })
}

/// GELU activation (Gaussian Error Linear Unit).
/// Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(a: &Tensor) -> Tensor {
    let sqrt_2_over_pi = (2.0f64 / std::f64::consts::PI).sqrt();
    a.map(move |v| {
        let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    })
}

/// GELU activation using the exact formula with erf.
pub fn gelu_exact(a: &Tensor) -> Tensor {
    let sqrt_half = (0.5f64).sqrt();
    a.map(move |v| {
        0.5 * v * (1.0 + erf_impl(v * sqrt_half))
    })
}

/// ReLU activation: max(0, x).
pub fn relu(a: &Tensor) -> Tensor {
    a.map(|v| if v > 0.0 { v } else { 0.0 })
}

/// ReLU6 activation: min(max(0, x), 6).
pub fn relu6(a: &Tensor) -> Tensor {
    a.map(|v| v.clamp(0.0, 6.0))
}

/// Leaky ReLU activation: max(alpha*x, x).
pub fn leaky_relu(a: &Tensor, alpha: f64) -> Tensor {
    a.map(move |v| if v > 0.0 { v } else { alpha * v })
}

/// Parametric ReLU activation (same as leaky_relu but alpha is learned).
pub fn prelu(a: &Tensor, alpha: f64) -> Tensor {
    leaky_relu(a, alpha)
}

/// ELU activation: x if x > 0 else alpha * (exp(x) - 1).
pub fn elu(a: &Tensor, alpha: f64) -> Tensor {
    a.map(move |v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
}

/// SELU activation: scale * elu(x, alpha).
/// where scale = 1.0507009873554804934193349852946 and alpha = 1.6732632423543772848170429916717
pub fn selu(a: &Tensor) -> Tensor {
    let scale = 1.0507009873554804934193349852946;
    let alpha = 1.6732632423543772848170429916717;
    a.map(move |v| {
        if v > 0.0 { scale * v } else { scale * alpha * (v.exp() - 1.0) }
    })
}

/// CELU activation: max(0, x) + min(0, alpha * (exp(x/alpha) - 1)).
pub fn celu(a: &Tensor, alpha: f64) -> Tensor {
    a.map(move |v| {
        if v > 0.0 { v } else { alpha * ((v / alpha).exp() - 1.0) }
    })
}

/// Softmax activation along dimension 0 (for 2D: along rows).
pub fn softmax(a: &Tensor, dim: usize) -> Tensor {
    assert!(dim < a.ndim(), "Dimension {} out of bounds", dim);
    let dim_size = a.shape()[dim];
    let sub_size: usize = if a.ndim() > 1 { a.shape()[dim + 1..].iter().product() } else { 1 };
    let outer_size: usize = a.shape()[..dim].iter().product();
    let mut data = a.data().to_vec();

    for outer in 0..outer_size {
        let base = outer * dim_size * sub_size;
        // Find max for numerical stability
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..dim_size {
            let idx = base + i * sub_size;
            for j in 0..sub_size {
                let v = data[idx + j];
                if v > max_val { max_val = v; }
            }
        }
        // Compute exp and sum
        let mut sum = 0.0;
        for i in 0..dim_size {
            let idx = base + i * sub_size;
            for j in 0..sub_size {
                let e = (data[idx + j] - max_val).exp();
                data[idx + j] = e;
                sum += e;
            }
        }
        // Normalize
        for i in 0..dim_size {
            let idx = base + i * sub_size;
            for j in 0..sub_size {
                data[idx + j] /= sum;
            }
        }
    }

    Tensor::new(data, a.shape().to_vec())
}

/// LogSoftmax activation (numerically stable).
pub fn log_softmax(a: &Tensor, dim: usize) -> Tensor {
    softmax(a, dim).map(|v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY })
}

// =============================================================================
// Error Functions
// =============================================================================

/// Error function: erf(x).
/// Uses Abramowitz and Stegun approximation (maximum error: 1.5e-7).
pub fn erf(a: &Tensor) -> Tensor {
    a.map(erf_impl)
}

/// Complementary error function: 1 - erf(x).
pub fn erfc(a: &Tensor) -> Tensor {
    a.map(|v| 1.0 - erf_impl(v))
}

/// Internal erf implementation using Abramowitz and Stegun formula 7.1.26.
fn erf_impl(x: f64) -> f64 {
    // Constants
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

// =============================================================================
// Additional Special Functions
// =============================================================================

/// Log Gamma function (Stirling's approximation for large x).
pub fn lgamma(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v <= 0.0 && v.fract() == 0.0 { return f64::INFINITY; }
        if v <= 0.0 { return f64::NAN; }
        lanczos_gamma(v).ln()
    })
}

/// Gamma function using Lanczos approximation.
fn lanczos_gamma(x: f64) -> f64 {
    let g = 7.0;
    let coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        let pi = std::f64::consts::PI;
        pi / ((pi * x).sin() * lanczos_gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = coef[0];
        for t in 1..coef.len() {
            a += coef[t] / (x as f64 + t as f64);
        }
        let t = x as f64 + g + 0.5;
        let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
        sqrt_2pi * t.powf(x as f64 + 0.5) * (-t).exp() * a
    }
}

/// Digamma function (psi): derivative of log-gamma.
pub fn digamma(a: &Tensor) -> Tensor {
    a.map(digamma_impl)
}

fn digamma_impl(x: f64) -> f64 {
    if x <= 0.0 { return f64::NAN; }
    let mut result = 0.0;
    let mut xx = x;
    while xx < 6.0 {
        result -= 1.0 / xx;
        xx += 1.0;
    }
    result += xx.ln() - 0.5 / xx;
    let x2 = 1.0 / (xx * xx);
    result -= x2 * (1.0 / 12.0 + x2 * (1.0 / 120.0 + x2 * (1.0 / 252.0)));
    result
}

/// Inverse of sigmoid function: logit(x) = log(x / (1 - x)).
pub fn logit(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v <= 0.0 || v >= 1.0 { f64::NAN } else { (v / (1.0 - v)).ln() }
    })
}

/// Softsign function: x / (1 + |x|).
pub fn softsign(a: &Tensor) -> Tensor {
    a.map(|v| v / (1.0 + v.abs()))
}

/// Inverse softsign: x / (1 - |x|).
pub fn inverse_softsign(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v.abs() >= 1.0 { f64::NAN } else { v / (1.0 - v.abs()) }
    })
}

/// Log-sigmoid: log(sigmoid(x)) = -softplus(-x).
pub fn log_sigmoid(a: &Tensor) -> Tensor {
    a.map(|v| {
        if v >= 0.0 {
            (-v).ln_1p()
        } else {
            v + (-v).exp().ln_1p()
        }
    })
}

/// Hard sigmoid: clamp(0.2 * x + 0.5, 0, 1).
pub fn hard_sigmoid(a: &Tensor) -> Tensor {
    a.map(|v| (0.2 * v + 0.5).clamp(0.0, 1.0))
}

/// Hard tanh: clamp(3 * x, -1, 1).
pub fn hard_tanh(a: &Tensor) -> Tensor {
    a.map(|v| (3.0 * v).clamp(-1.0, 1.0))
}

/// Hard swish: x * hard_sigmoid(x).
pub fn hard_swish(a: &Tensor) -> Tensor {
    a.map(|v| v * (0.2 * v + 0.5).clamp(0.0, 1.0))
}

/// Squared ReLU: x^2 for x > 0, else 0.
pub fn squared_relu(a: &Tensor) -> Tensor {
    a.map(|v| if v > 0.0 { v * v } else { 0.0 })
}

/// Bent identity: (sqrt(x^2 + 1) - 1) / 2 + x.
pub fn bent_identity(a: &Tensor) -> Tensor {
    a.map(|v| (v * v + 1.0).sqrt() - 1.0) / 2.0 + a
}

/// SiLU as a separate function (same as swish with beta=1).
pub fn swish_default(a: &Tensor) -> Tensor {
    silu(a)
}

/// GLU activation: x * sigmoid(Wx + b).
/// Simplified: element-wise x * sigmoid(x).
pub fn glu(a: &Tensor) -> Tensor {
    a.map(|v| {
        let sig = if v >= 0.0 { 1.0 / (1.0 + (-v).exp()) }
        else { v.exp() / (1.0 + v.exp()) };
        v * sig
    })
}

/// SwiGLU: Swish(x) * (Wx + b).
/// Simplified: x * sigmoid(x) * x = x^2 * sigmoid(x).
pub fn swiglu(a: &Tensor) -> Tensor {
    a.map(|v| {
        let sig = if v >= 0.0 { 1.0 / (1.0 + (-v).exp()) }
        else { v.exp() / (1.0 + v.exp()) };
        v * v * sig
    })
}

/// GeLU variant with tanh approximation (fast).
pub fn gelu_fast(a: &Tensor) -> Tensor {
    let k = 0.7978845608028654; // sqrt(2/pi)
    a.map(move |v| {
        0.5 * v * (1.0 + (k * v * (1.0 + 0.044715 * v * v)).tanh())
    })
}

// =============================================================================
// Interpolation and Combination Functions
// =============================================================================

/// Smoothstep function: Hermite interpolation between edge0 and edge1.
pub fn smoothstep(edge0: f64, edge1: f64, a: &Tensor) -> Tensor {
    assert!(edge0 < edge1);
    a.map(move |v| {
        let t = ((v - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    })
}

/// Smootherstep function.
pub fn smootherstep(edge0: f64, edge1: f64, a: &Tensor) -> Tensor {
    assert!(edge0 < edge1);
    a.map(move |v| {
        let t = ((v - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    })
}

/// Step function: 1 if x > threshold else 0.
pub fn step(a: &Tensor, threshold: f64) -> Tensor {
    a.map(move |v| if v > threshold { 1.0 } else { 0.0 })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp() {
        let a = Tensor::from_slice(&[0.0, 1.0, 2.0], vec![3]);
        let b = exp(&a);
        assert!((b.get(0) - 1.0).abs() < 1e-10);
        assert!((b.get(1) - std::f64::consts::E).abs() < 1e-10);
        assert!((b.get(2) - (2.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_log() {
        let a = Tensor::from_slice(&[1.0, std::f64::consts::E, std::f64::consts::E.powi(2)], vec![3]);
        let b = log(&a);
        assert!((b.get(0) - 0.0).abs() < 1e-10);
        assert!((b.get(1) - 1.0).abs() < 1e-10);
        assert!((b.get(2) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_negative() {
        let a = Tensor::from_slice(&[-1.0, 0.0], vec![2]);
        let b = log(&a);
        assert!(b.get(0).is_nan());
        assert!(b.get(1).is_nan());
    }

    #[test]
    fn test_log1p() {
        let a = Tensor::from_slice(&[0.0, 1e-15, 1.0], vec![3]);
        let b = log1p(&a);
        assert!((b.get(0) - 0.0).abs() < 1e-10);
        assert!(!b.get(1).is_nan()); // Should not lose precision for small values
    }

    #[test]
    fn test_sin() {
        let a = Tensor::from_slice(&[0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI], vec![3]);
        let b = sin(&a);
        assert!(b.get(0).abs() < 1e-10);
        assert!((b.get(1) - 1.0).abs() < 1e-10);
        assert!(b.get(2).abs() < 1e-10);
    }

    #[test]
    fn test_cos() {
        let a = Tensor::from_slice(&[0.0, std::f64::consts::PI], vec![2]);
        let b = cos(&a);
        assert!((b.get(0) - 1.0).abs() < 1e-10);
        assert!((b.get(1) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_tan() {
        let a = Tensor::from_slice(&[0.0, std::f64::consts::PI / 4.0], vec![2]);
        let b = tan(&a);
        assert!(b.get(0).abs() < 1e-10);
        assert!((b.get(1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_asin() {
        let a = Tensor::from_slice(&[0.0, 0.5, 1.0], vec![3]);
        let b = asin(&a);
        assert!(b.get(0).abs() < 1e-10);
        assert!((b.get(2) - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_asin_out_of_range() {
        let a = Tensor::from_slice(&[2.0], vec![1]);
        let b = asin(&a);
        assert!(b.get(0).is_nan());
    }

    #[test]
    fn test_atan() {
        let a = Tensor::from_slice(&[0.0, 1.0], vec![2]);
        let b = atan(&a);
        assert!(b.get(0).abs() < 1e-10);
        assert!((b.get(1) - std::f64::consts::PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_atan2() {
        let y = Tensor::from_slice(&[1.0, -1.0], vec![2]);
        let x = Tensor::from_slice(&[1.0, -1.0], vec![2]);
        let a = atan2(&y, &x);
        assert!((a.get(0) - std::f64::consts::PI / 4.0).abs() < 1e-10);
        assert!((a.get(1) - (-3.0 * std::f64::consts::PI / 4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sinh() {
        let a = Tensor::from_slice(&[0.0, 1.0], vec![2]);
        let b = sinh(&a);
        assert!(b.get(0).abs() < 1e-10);
        assert!((b.get(1) - ((1.0_f64).sinh())).abs() < 1e-10);
    }

    #[test]
    fn test_cosh() {
        let a = Tensor::from_slice(&[0.0, 1.0], vec![2]);
        let b = cosh(&a);
        assert!((b.get(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tanh() {
        let a = Tensor::from_slice(&[0.0, 1.0, 100.0], vec![3]);
        let b = tanh(&a);
        assert!(b.get(0).abs() < 1e-10);
        assert!(b.get(2) < 1.0 && b.get(2) > 0.99);
    }

    #[test]
    fn test_asinh() {
        let a = Tensor::from_slice(&[0.0, 1.0], vec![2]);
        let b = asinh(&a);
        assert!(b.get(0).abs() < 1e-10);
    }

    #[test]
    fn test_acosh() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let b = acosh(&a);
        assert!(b.get(0).abs() < 1e-10);
    }

    #[test]
    fn test_acosh_invalid() {
        let a = Tensor::from_slice(&[0.5], vec![1]);
        let b = acosh(&a);
        assert!(b.get(0).is_nan());
    }

    #[test]
    fn test_atanh() {
        let a = Tensor::from_slice(&[0.0, 0.5], vec![2]);
        let b = atanh(&a);
        assert!(b.get(0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt() {
        let a = Tensor::from_slice(&[0.0, 1.0, 4.0, 9.0], vec![4]);
        let b = sqrt(&a);
        assert!((b.get(0) - 0.0).abs() < 1e-10);
        assert!((b.get(1) - 1.0).abs() < 1e-10);
        assert!((b.get(2) - 2.0).abs() < 1e-10);
        assert!((b.get(3) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_negative() {
        let a = Tensor::from_slice(&[-1.0], vec![1]);
        let b = sqrt(&a);
        assert!(b.get(0).is_nan());
    }

    #[test]
    fn test_cbrt() {
        let a = Tensor::from_slice(&[-8.0, -1.0, 0.0, 1.0, 8.0, 27.0], vec![6]);
        let b = cbrt(&a);
        assert!((b.get(0) - (-2.0)).abs() < 1e-10);
        assert!((b.get(1) - (-1.0)).abs() < 1e-10);
        assert!((b.get(3) - 1.0).abs() < 1e-10);
        assert!((b.get(4) - 2.0).abs() < 1e-10);
        assert!((b.get(5) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_abs() {
        let a = Tensor::from_slice(&[-3.0, 0.0, 5.0], vec![3]);
        let b = abs(&a);
        assert_eq!(b.get(0), 3.0);
        assert_eq!(b.get(1), 0.0);
        assert_eq!(b.get(2), 5.0);
    }

    #[test]
    fn test_sign() {
        let a = Tensor::from_slice(&[-5.0, 0.0, 3.0], vec![3]);
        let b = sign(&a);
        assert_eq!(b.get(0), -1.0);
        assert_eq!(b.get(1), 0.0);
        assert_eq!(b.get(2), 1.0);
    }

    #[test]
    fn test_ceil() {
        let a = Tensor::from_slice(&[1.1, 2.5, -1.5], vec![3]);
        let b = ceil(&a);
        assert_eq!(b.get(0), 2.0);
        assert_eq!(b.get(1), 3.0);
        assert_eq!(b.get(2), -1.0);
    }

    #[test]
    fn test_floor() {
        let a = Tensor::from_slice(&[1.1, 2.5, -1.5], vec![3]);
        let b = floor(&a);
        assert_eq!(b.get(0), 1.0);
        assert_eq!(b.get(1), 2.0);
        assert_eq!(b.get(2), -2.0);
    }

    #[test]
    fn test_round() {
        let a = Tensor::from_slice(&[1.4, 1.5, 2.5, -1.5], vec![4]);
        let b = round(&a);
        assert_eq!(b.get(0), 1.0);
        assert_eq!(b.get(1), 2.0);
        assert_eq!(b.get(2), 2.0);
        assert_eq!(b.get(3), -2.0);
    }

    #[test]
    fn test_trunc() {
        let a = Tensor::from_slice(&[1.9, -1.9, 0.5], vec![3]);
        let b = trunc(&a);
        assert_eq!(b.get(0), 1.0);
        assert_eq!(b.get(1), -1.0);
    }

    #[test]
    fn test_fract() {
        let a = Tensor::from_slice(&[3.14, -1.5, 2.0], vec![3]);
        let b = fract(&a);
        assert!((b.get(0) - 0.14).abs() < 1e-10);
        assert!((b.get(1) - 0.5).abs() < 1e-10);
        assert_eq!(b.get(2), 0.0);
    }

    #[test]
    fn test_clamp() {
        let a = Tensor::from_slice(&[-1.0, 0.5, 2.0], vec![3]);
        let b = clamp(&a, 0.0, 1.0);
        assert_eq!(b.get(0), 0.0);
        assert!((b.get(1) - 0.5).abs() < 1e-10);
        assert_eq!(b.get(2), 1.0);
    }

    #[test]
    fn test_isnan() {
        let a = Tensor::from_slice(&[1.0, f64::NAN, 3.0], vec![3]);
        let b = isnan(&a);
        assert_eq!(b.get(0), 0.0);
        assert_eq!(b.get(1), 1.0);
        assert_eq!(b.get(2), 0.0);
    }

    #[test]
    fn test_isinf() {
        let a = Tensor::from_slice(&[1.0, f64::INFINITY, f64::NEG_INFINITY], vec![3]);
        let b = isinf(&a);
        assert_eq!(b.get(0), 0.0);
        assert_eq!(b.get(1), 1.0);
        assert_eq!(b.get(2), 1.0);
    }

    #[test]
    fn test_isfinite() {
        let a = Tensor::from_slice(&[1.0, f64::NAN, f64::INFINITY], vec![3]);
        let b = isfinite(&a);
        assert_eq!(b.get(0), 1.0);
        assert_eq!(b.get(1), 0.0);
        assert_eq!(b.get(2), 0.0);
    }

    #[test]
    fn test_sigmoid() {
        let a = Tensor::from_slice(&[0.0, 100.0, -100.0], vec![3]);
        let b = sigmoid(&a);
        assert!((b.get(0) - 0.5).abs() < 1e-10);
        assert!((b.get(1) - 1.0).abs() < 1e-10);
        assert!((b.get(2) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_relu() {
        let a = Tensor::from_slice(&[-2.0, 0.0, 3.0], vec![3]);
        let b = relu(&a);
        assert_eq!(b.get(0), 0.0);
        assert_eq!(b.get(1), 0.0);
        assert_eq!(b.get(2), 3.0);
    }

    #[test]
    fn test_leaky_relu() {
        let a = Tensor::from_slice(&[-2.0, 0.0, 3.0], vec![3]);
        let b = leaky_relu(&a, 0.1);
        assert!((b.get(0) - (-0.2)).abs() < 1e-10);
        assert_eq!(b.get(1), 0.0);
        assert_eq!(b.get(2), 3.0);
    }

    #[test]
    fn test_gelu() {
        let a = Tensor::from_slice(&[0.0, 1.0, -1.0], vec![3]);
        let b = gelu(&a);
        assert!(b.get(0).abs() < 1e-10); // gelu(0) = 0
        assert!(b.get(1) > 0.5 && b.get(1) < 1.0);
        assert!(b.get(2) < 0.0 && b.get(2) > -0.5);
    }

    #[test]
    fn test_softplus() {
        let a = Tensor::from_slice(&[0.0, 10.0, -10.0], vec![3]);
        let b = softplus(&a);
        assert!((b.get(0) - (2.0_f64).ln()).abs() < 1e-10); // ln(1 + e^0) = ln(2)
        assert!((b.get(1) - 10.0).abs() < 1e-10);
        assert!(b.get(2).abs() < 1e-10);
    }

    #[test]
    fn test_mish() {
        let a = Tensor::from_slice(&[0.0, 1.0, -1.0], vec![3]);
        let b = mish(&a);
        assert!(b.get(0).abs() < 1e-10);
        assert!(b.get(1) > 0.5);
    }

    #[test]
    fn test_silu() {
        let a = Tensor::from_slice(&[0.0, 1.0, -1.0], vec![3]);
        let b = silu(&a);
        assert!(b.get(0).abs() < 1e-10);
        assert!(b.get(1) > 0.5);
        assert!(b.get(2) < 0.0);
    }

    #[test]
    fn test_elu() {
        let a = Tensor::from_slice(&[-2.0, 0.0, 3.0], vec![3]);
        let b = elu(&a, 1.0);
        assert!(b.get(0) < 0.0); // alpha * (exp(-2) - 1)
        assert_eq!(b.get(1), 0.0);
        assert_eq!(b.get(2), 3.0);
    }

    #[test]
    fn test_erf() {
        let a = Tensor::from_slice(&[0.0, 1.0, -1.0], vec![3]);
        let b = erf(&a);
        assert!(b.get(0).abs() < 1e-7);
        assert!((b.get(1) - 0.8427007929497149).abs() < 1e-7);
        assert!((b.get(2) + 0.8427007929497149).abs() < 1e-7);
    }

    #[test]
    fn test_erfc() {
        let a = Tensor::from_slice(&[0.0], vec![1]);
        let b = erfc(&a);
        assert!((b.get(0) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let a = Tensor::from_slice(&[-5.0, -2.0, 0.0, 2.0, 5.0], vec![5]);
        let b = sigmoid(&a);
        for i in 0..5 {
            let j = 4 - i;
            assert!((b.get(i) + b.get(j) - 1.0).abs() < 1e-10,
                "Sigmoid not symmetric: sigmoid({}) + sigmoid({}) = {} + {} = {}",
                a.get(i), a.get(j), b.get(i), b.get(j), b.get(i) + b.get(j));
        }
    }

    #[test]
    fn test_tanh_properties() {
        let a = Tensor::from_slice(&[-10.0, 0.0, 10.0], vec![3]);
        let b = tanh(&a);
        assert!(b.get(0) > -1.0 && b.get(0) < 0.0);
        assert!(b.get(1).abs() < 1e-10);
        assert!(b.get(2) > 0.0 && b.get(2) < 1.0);
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = softmax(&a, 0);
        let sum: f64 = (0..4).map(|i| b.get(i)).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_less() {
        let a = Tensor::from_slice(&[1.0, 3.0], vec![2]);
        let b = Tensor::from_slice(&[2.0, 2.0], vec![2]);
        let c = less(&a, &b);
        assert_eq!(c.get(0), 1.0);
        assert_eq!(c.get(1), 0.0);
    }

    #[test]
    fn test_equal_fn() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let b = Tensor::from_slice(&[1.0, 3.0], vec![2]);
        let c = equal(&a, &b);
        assert_eq!(c.get(0), 1.0);
        assert_eq!(c.get(1), 0.0);
    }

    #[test]
    fn test_logit() {
        let a = Tensor::from_slice(&[0.25, 0.5, 0.75], vec![3]);
        let b = logit(&a);
        assert!((b.get(1)).abs() < 1e-10); // logit(0.5) = 0
        assert!(b.get(0) < 0.0);
        assert!(b.get(2) > 0.0);
    }

    #[test]
    fn test_hard_sigmoid() {
        let a = Tensor::from_slice(&[-10.0, 0.0, 10.0], vec![3]);
        let b = hard_sigmoid(&a);
        assert_eq!(b.get(0), 0.0);
        assert!((b.get(1) - 0.5).abs() < 1e-10);
        assert_eq!(b.get(2), 1.0);
    }

    #[test]
    fn test_lgamma() {
        let a = Tensor::from_slice(&[1.0, 2.0, 5.0], vec![3]);
        let b = lgamma(&a);
        assert!(b.get(0).abs() < 1e-10); // lgamma(1) = 0
        assert!((b.get(1) - 0.0).abs() < 1e-10); // lgamma(2) = 0
    }

    #[test]
    fn test_softsign() {
        let a = Tensor::from_slice(&[-10.0, 0.0, 10.0], vec![3]);
        let b = softsign(&a);
        assert!(b.get(0) > -1.0 && b.get(0) < 0.0);
        assert!(b.get(1).abs() < 1e-10);
        assert!(b.get(2) > 0.0 && b.get(2) < 1.0);
    }

    #[test]
    fn test_pow() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![3]);
        let b = pow(&a, 3.0);
        assert_eq!(b.get(0), 8.0);
        assert_eq!(b.get(1), 27.0);
        assert_eq!(b.get(2), 64.0);
    }

    #[test]
    fn test_round_to() {
        let a = Tensor::from_slice(&[1.2345, 2.3456], vec![2]);
        let b = round_to(&a, 2);
        assert!((b.get(0) - 1.23).abs() < 1e-10);
        assert!((b.get(1) - 2.35).abs() < 1e-10);
    }

    #[test]
    fn test_where_tensor() {
        let cond = Tensor::from_slice(&[1.0, 0.0, 1.0], vec![3]);
        let val = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![3]);
        let other = Tensor::from_slice(&[-10.0, -20.0, -30.0], vec![3]);
        let c = where_tensor(&cond, &val, &other);
        assert_eq!(c.get(0), 10.0);
        assert_eq!(c.get(1), -20.0);
        assert_eq!(c.get(2), 30.0);
    }

    #[test]
    fn test_exp2() {
        let a = Tensor::from_slice(&[0.0, 1.0, 2.0, 10.0], vec![4]);
        let b = exp2(&a);
        assert!((b.get(0) - 1.0).abs() < 1e-10);
        assert!((b.get(1) - 2.0).abs() < 1e-10);
        assert!((b.get(2) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gelu_exact_approx_agree() {
        let a = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let b_approx = gelu(&a);
        let b_exact = gelu_exact(&a);
        for i in 0..5 {
            assert!((b_approx.get(i) - b_exact.get(i)).abs() < 0.01,
                "GELU approx and exact differ at {}: {} vs {}", i, b_approx.get(i), b_exact.get(i));
        }
    }
}
