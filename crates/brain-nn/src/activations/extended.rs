//! # Extended Neural Activation Functions
//!
//! Production-grade implementations of modern activations:
//! - `ELU` (Exponential Linear Unit)
//! - `CELU` (Continuously Differentiable Exponential Linear Unit)
//! - `SELU` (Scaled Exponential Linear Unit with self-normalizing fixed point)
//! - `Softplus` (Smooth approximation to ReLU: ln(1 + e^(beta * x)) / beta)
//! - `Softsign` (x / (1 + |x|))
//! - `HardSigmoid` (clamp((x + 3) / 6, 0, 1))
//! - `HardSwish` (x * HardSigmoid(x))
//! - `HardTanh` (clamp(x, min, max))
//! - `GLU` (Gated Linear Unit: a * sigmoid(b))
//! - `SwiGLU` (Swish Gated Linear Unit: swish(a) * b)

use crate::activations::Activation;
use brain_core::Tensor;

/// Exponential Linear Unit: alpha * (exp(x) - 1) for x <= 0, x for x > 0.
#[derive(Debug, Clone)]
pub struct ELU {
    pub alpha: f64,
}

impl Default for ELU {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl ELU {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Activation for ELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let alpha = self.alpha;
        input.map(move |x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
    }
}

/// Computes ELU elementwise.
pub fn elu(input: &Tensor, alpha: f64) -> Tensor {
    ELU::new(alpha).forward(input)
}

/// Continuously Differentiable Exponential Linear Unit.
#[derive(Debug, Clone)]
pub struct CELU {
    pub alpha: f64,
}

impl Default for CELU {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl CELU {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0, "CELU alpha must be positive");
        Self { alpha }
    }
}

impl Activation for CELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let alpha = self.alpha;
        input.map(move |x| {
            if x > 0.0 {
                x
            } else {
                alpha * ((x / alpha).exp() - 1.0)
            }
        })
    }
}

/// Computes CELU elementwise.
pub fn celu(input: &Tensor, alpha: f64) -> Tensor {
    CELU::new(alpha).forward(input)
}

/// Scaled Exponential Linear Unit (Klambauer et al., 2017).
/// Induces self-normalizing properties with mean 0 and variance 1 across deep networks.
#[derive(Debug, Clone, Copy, Default)]
pub struct SELU;

pub const SELU_SCALE: f64 = 1.0507009873554804934193349852946;
pub const SELU_ALPHA: f64 = 1.6732632423543772848170429916717;

impl Activation for SELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.map(|x| {
            if x > 0.0 {
                SELU_SCALE * x
            } else {
                SELU_SCALE * SELU_ALPHA * (x.exp() - 1.0)
            }
        })
    }
}

/// Computes SELU elementwise.
pub fn selu(input: &Tensor) -> Tensor {
    SELU.forward(input)
}

/// Softplus activation: ln(1 + exp(beta * x)) / beta.
#[derive(Debug, Clone)]
pub struct Softplus {
    pub beta: f64,
    pub threshold: f64,
}

impl Default for Softplus {
    fn default() -> Self {
        Self {
            beta: 1.0,
            threshold: 20.0,
        }
    }
}

impl Softplus {
    pub fn new(beta: f64, threshold: f64) -> Self {
        Self { beta, threshold }
    }
}

impl Activation for Softplus {
    fn forward(&self, input: &Tensor) -> Tensor {
        let beta = self.beta;
        let threshold = self.threshold;
        input.map(move |x| {
            if x * beta > threshold {
                x
            } else {
                (1.0 + (beta * x).exp()).ln() / beta
            }
        })
    }
}

/// Computes Softplus elementwise.
pub fn softplus(input: &Tensor, beta: f64) -> Tensor {
    Softplus::new(beta, 20.0).forward(input)
}

/// Softsign activation: x / (1 + |x|).
#[derive(Debug, Clone, Copy, Default)]
pub struct Softsign;

impl Activation for Softsign {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.map(|x| x / (1.0 + x.abs()))
    }
}

/// Computes Softsign elementwise.
pub fn softsign(input: &Tensor) -> Tensor {
    Softsign.forward(input)
}

/// HardSigmoid activation: clamp((x + 3) / 6, 0, 1).
#[derive(Debug, Clone, Copy, Default)]
pub struct HardSigmoid;

impl Activation for HardSigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.map(|x| ((x + 3.0) / 6.0).clamp(0.0, 1.0))
    }
}

/// Computes HardSigmoid elementwise.
pub fn hard_sigmoid(input: &Tensor) -> Tensor {
    HardSigmoid.forward(input)
}

/// HardSwish activation: x * HardSigmoid(x).
#[derive(Debug, Clone, Copy, Default)]
pub struct HardSwish;

impl Activation for HardSwish {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.map(|x| x * ((x + 3.0) / 6.0).clamp(0.0, 1.0))
    }
}

/// Computes HardSwish elementwise.
pub fn hard_swish(input: &Tensor) -> Tensor {
    HardSwish.forward(input)
}

/// HardTanh activation: clamp(x, min_val, max_val).
#[derive(Debug, Clone)]
pub struct HardTanh {
    pub min_val: f64,
    pub max_val: f64,
}

impl Default for HardTanh {
    fn default() -> Self {
        Self {
            min_val: -1.0,
            max_val: 1.0,
        }
    }
}

impl Activation for HardTanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (min, max) = (self.min_val, self.max_val);
        input.map(move |x| x.clamp(min, max))
    }
}

/// Gated Linear Unit (GLU): splits input into two halves along `dim`, computes a * sigmoid(b).
pub fn glu(input: &Tensor, dim: usize) -> Tensor {
    assert!(dim < input.ndim(), "glu: dim out of bounds");
    assert_eq!(
        input.shape()[dim] % 2,
        0,
        "glu: dimension size must be divisible by 2"
    );

    let chunks = input.chunk(2, dim);
    let a = &chunks[0];
    let b = &chunks[1];
    let sig_b = b.map(|x| 1.0 / (1.0 + (-x).exp()));
    a.map2(&sig_b, |x, y| x * y)
}

/// Swish Gated Linear Unit (SwiGLU): splits input into two halves, computes Swish(a) * b.
pub fn swiglu(input: &Tensor, dim: usize) -> Tensor {
    assert!(dim < input.ndim(), "swiglu: dim out of bounds");
    assert_eq!(
        input.shape()[dim] % 2,
        0,
        "swiglu: dimension size must be divisible by 2"
    );

    let chunks = input.chunk(2, dim);
    let a = &chunks[0];
    let b = &chunks[1];
    let swish_a = a.map(|x| x / (1.0 + (-x).exp()));
    swish_a.map2(b, |x, y| x * y)
}
