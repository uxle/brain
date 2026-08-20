//! Mathematical functions and neural network activation functions for tensors.
//!
//! This module provides element-wise transcendentals (exp, log, sqrt, cbrt),
//! trigonometry (sin, cos, tan, asin, acos, atan, sinc), hyperbolic functions (sinh, cosh, tanh),
//! rounding/clamping, and activation functions (sigmoid, relu, gelu, silu, elu, selu, mish, softplus).

use crate::tensor::Tensor;

// =============================================================================
// Transcendentals & Exponentials
// =============================================================================

/// Element-wise natural exponential: e^x.
pub fn exp(a: &Tensor) -> Tensor {
    a.map(|x| x.exp())
}

/// Element-wise base-2 exponential: 2^x.
pub fn exp2(a: &Tensor) -> Tensor {
    a.map(|x| x.exp2())
}

/// Element-wise e^x - 1 (accurate for small x).
pub fn expm1(a: &Tensor) -> Tensor {
    a.map(|x| x.exp_m1())
}

/// Element-wise natural logarithm: ln(x).
pub fn log(a: &Tensor) -> Tensor {
    a.map(|x| x.ln())
}

/// Element-wise base-2 logarithm: log2(x).
pub fn log2(a: &Tensor) -> Tensor {
    a.map(|x| x.log2())
}

/// Element-wise base-10 logarithm: log10(x).
pub fn log10(a: &Tensor) -> Tensor {
    a.map(|x| x.log10())
}

/// Element-wise ln(1 + x) (accurate for small x).
pub fn log1p(a: &Tensor) -> Tensor {
    a.map(|x| x.ln_1p())
}

/// Element-wise square root: sqrt(x).
pub fn sqrt(a: &Tensor) -> Tensor {
    a.map(|x| x.sqrt())
}

/// Element-wise reciprocal square root: 1 / sqrt(x).
pub fn rsqrt(a: &Tensor) -> Tensor {
    a.map(|x| 1.0 / x.sqrt())
}

/// Element-wise cube root: cbrt(x).
pub fn cbrt(a: &Tensor) -> Tensor {
    a.map(|x| x.cbrt())
}

// =============================================================================
// Trigonometric & Hyperbolic Functions
// =============================================================================

/// Element-wise sine: sin(x).
pub fn sin(a: &Tensor) -> Tensor {
    a.map(|x| x.sin())
}

/// Element-wise cosine: cos(x).
pub fn cos(a: &Tensor) -> Tensor {
    a.map(|x| x.cos())
}

/// Element-wise tangent: tan(x).
pub fn tan(a: &Tensor) -> Tensor {
    a.map(|x| x.tan())
}

/// Element-wise arcsine: asin(x).
pub fn asin(a: &Tensor) -> Tensor {
    a.map(|x| x.asin())
}

/// Element-wise arccosine: acos(x).
pub fn acos(a: &Tensor) -> Tensor {
    a.map(|x| x.acos())
}

/// Element-wise arctangent: atan(x).
pub fn atan(a: &Tensor) -> Tensor {
    a.map(|x| x.atan())
}

/// Element-wise hyperbolic sine: sinh(x).
pub fn sinh(a: &Tensor) -> Tensor {
    a.map(|x| x.sinh())
}

/// Element-wise hyperbolic cosine: cosh(x).
pub fn cosh(a: &Tensor) -> Tensor {
    a.map(|x| x.cosh())
}

/// Element-wise hyperbolic tangent: tanh(x).
pub fn tanh(a: &Tensor) -> Tensor {
    a.map(|x| x.tanh())
}

/// Element-wise inverse hyperbolic sine: asinh(x).
pub fn asinh(a: &Tensor) -> Tensor {
    a.map(|x| x.asinh())
}

/// Element-wise inverse hyperbolic cosine: acosh(x).
pub fn acosh(a: &Tensor) -> Tensor {
    a.map(|x| x.acosh())
}

/// Element-wise inverse hyperbolic tangent: atanh(x).
pub fn atanh(a: &Tensor) -> Tensor {
    a.map(|x| x.atanh())
}

// =============================================================================
// Rounding, Signs & Clamping
// =============================================================================

/// Element-wise absolute value: |x|.
pub fn abs(a: &Tensor) -> Tensor {
    a.map(|x| x.abs())
}

/// Element-wise signum: -1.0, 0.0, or 1.0.
pub fn signum(a: &Tensor) -> Tensor {
    a.map(|x| x.signum())
}

/// Element-wise sign: -1.0, 0.0, or 1.0 (alias of `signum`).
pub fn sign(a: &Tensor) -> Tensor {
    signum(a)
}

/// Element-wise reciprocal: 1 / x.
pub fn recip(a: &Tensor) -> Tensor {
    a.map(|x| 1.0 / x)
}

/// Element-wise square: x * x.
pub fn square(a: &Tensor) -> Tensor {
    a.map(|x| x * x)
}

/// Element-wise floor: floor(x).
pub fn floor(a: &Tensor) -> Tensor {
    a.map(|x| x.floor())
}

/// Element-wise ceiling: ceil(x).
pub fn ceil(a: &Tensor) -> Tensor {
    a.map(|x| x.ceil())
}

/// Element-wise round to nearest integer.
pub fn round(a: &Tensor) -> Tensor {
    a.map(|x| x.round())
}

/// Element-wise truncation to integer.
pub fn trunc(a: &Tensor) -> Tensor {
    a.map(|x| x.trunc())
}

/// Element-wise fractional part: x - trunc(x).
pub fn frac(a: &Tensor) -> Tensor {
    a.map(|x| x.fract())
}

/// Element-wise clamp between `min_val` and `max_val`.
pub fn clamp(a: &Tensor, min_val: f64, max_val: f64) -> Tensor {
    a.map(|x| x.max(min_val).min(max_val))
}

/// Alias for `clamp`.
pub fn clip(a: &Tensor, min_val: f64, max_val: f64) -> Tensor {
    clamp(a, min_val, max_val)
}

// =============================================================================
// Activation Functions
// =============================================================================

/// Sigmoid activation: 1 / (1 + e^-x).
pub fn sigmoid(a: &Tensor) -> Tensor {
    a.map(|x| 1.0 / (1.0 + (-x).exp()))
}

/// Log-sigmoid activation: ln(1 / (1 + e^-x)).
pub fn log_sigmoid(a: &Tensor) -> Tensor {
    a.map(|x| {
        if x >= 0.0 {
            -(-x).exp().ln_1p()
        } else {
            x - x.exp().ln_1p()
        }
    })
}

/// Rectified Linear Unit (ReLU): max(0, x).
pub fn relu(a: &Tensor) -> Tensor {
    a.map(|x| x.max(0.0))
}

/// ReLU6 activation: min(max(0, x), 6).
pub fn relu6(a: &Tensor) -> Tensor {
    a.map(|x| x.max(0.0).min(6.0))
}

/// Leaky ReLU activation with negative slope.
pub fn leaky_relu(a: &Tensor, negative_slope: f64) -> Tensor {
    a.map(|x| if x >= 0.0 { x } else { x * negative_slope })
}

/// Exponential Linear Unit (ELU).
pub fn elu(a: &Tensor, alpha: f64) -> Tensor {
    a.map(|x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) })
}

/// Scaled Exponential Linear Unit (SELU).
pub fn selu(a: &Tensor) -> Tensor {
    const SCALE: f64 = 1.0507009873554804934193349852946;
    const ALPHA: f64 = 1.6732632423543772848170429916717;
    a.map(|x| SCALE * if x >= 0.0 { x } else { ALPHA * (x.exp() - 1.0) })
}

/// Continuously Differentiable Exponential Linear Unit (CELU).
pub fn celu(a: &Tensor, alpha: f64) -> Tensor {
    assert!(alpha != 0.0, "CELU alpha cannot be zero");
    a.map(|x| {
        if x >= 0.0 {
            x
        } else {
            alpha * ((x / alpha).exp() - 1.0)
        }
    })
}

/// Softplus activation: ln(1 + e^x).
pub fn softplus(a: &Tensor) -> Tensor {
    a.map(|x| {
        if x > 20.0 {
            x
        } else if x < -20.0 {
            x.exp()
        } else {
            (1.0 + x.exp()).ln()
        }
    })
}

/// Softsign activation: x / (1 + |x|).
pub fn softsign(a: &Tensor) -> Tensor {
    a.map(|x| x / (1.0 + x.abs()))
}

/// Hard sigmoid: clamp((x + 3) / 6, 0, 1).
pub fn hard_sigmoid(a: &Tensor) -> Tensor {
    a.map(|x| ((x + 3.0) / 6.0).max(0.0).min(1.0))
}

/// Hard swish: x * hard_sigmoid(x).
pub fn hard_swish(a: &Tensor) -> Tensor {
    a.map(|x| x * ((x + 3.0) / 6.0).max(0.0).min(1.0))
}

/// Hard tanh: clamp(x, -1, 1).
pub fn hard_tanh(a: &Tensor) -> Tensor {
    clamp(a, -1.0, 1.0)
}

/// Gaussian Error Linear Unit (GELU) with tanh approximation.
pub fn gelu(a: &Tensor) -> Tensor {
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    a.map(|x| 0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3))).tanh()))
}

/// Sigmoid Linear Unit (SiLU / Swish): x * sigmoid(x).
pub fn silu(a: &Tensor) -> Tensor {
    a.map(|x| x / (1.0 + (-x).exp()))
}

/// Mish activation: x * tanh(softplus(x)).
pub fn mish(a: &Tensor) -> Tensor {
    a.map(|x| {
        let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        x * sp.tanh()
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcendentals() {
        let a = Tensor::from_slice(&[0.0, 1.0, 2.0], vec![3]);
        assert_eq!(exp(&a).get(0), 1.0);
        assert_eq!(log(&exp(&a)).get(1), 1.0);
        assert_eq!(sqrt(&Tensor::full(vec![1], 4.0)).get(0), 2.0);
    }

    #[test]
    fn test_trig_and_hyperbolic() {
        let zero = Tensor::zeros(vec![1]);
        assert_eq!(sin(&zero).get(0), 0.0);
        assert_eq!(cos(&zero).get(0), 1.0);
        assert_eq!(tanh(&zero).get(0), 0.0);
    }

    #[test]
    fn test_activations() {
        let zero = Tensor::zeros(vec![1]);
        assert_eq!(sigmoid(&zero).get(0), 0.5);
        assert_eq!(
            relu(&Tensor::from_slice(&[-2.0, 3.0], vec![2])).data(),
            &[0.0, 3.0]
        );
        assert_eq!(gelu(&zero).get(0), 0.0);
        assert_eq!(silu(&zero).get(0), 0.0);
    }

    #[test]
    fn test_elementwise_math_ops() {
        let a = Tensor::from_slice(&[-2.0, 0.0, 4.0], vec![3]);
        assert_eq!(abs(&a).to_vec(), vec![2.0, 0.0, 4.0]);
        assert_eq!(clamp(&a, -1.0, 1.0).to_vec(), vec![-1.0, 0.0, 1.0]);

        let p = Tensor::from_slice(&[1.0, 4.0, 9.0], vec![3]);
        assert_eq!(sqrt(&p).to_vec(), vec![1.0, 2.0, 3.0]);
    }
}
