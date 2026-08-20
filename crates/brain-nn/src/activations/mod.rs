//! # Activation Functions & Registry
//!
//! Non-linear point activations: ReLU, LeakyReLU, Sigmoid, Tanh, GELU, FastGELU, Softmax, Swish, Mish.
#![allow(missing_docs)]

pub mod extended;
pub mod extra;
pub mod gelu;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod swish;

pub use extended::{
    celu, elu, glu, hard_sigmoid, hard_swish, selu, softplus, softsign, swiglu, HardSigmoid,
    HardSwish, HardTanh, Softplus, Softsign, CELU, ELU, SELU,
};
pub use extra::{
    hard_shrink, log_sigmoid, prelu, quiet_softmax, relu6, shrink, soft_shrink, softmin,
    tanh_shrink, threshold, thresholded_relu, HardShrink, LogSigmoid, PReLU, QuietSoftmax, ReLU6,
    Shrink, SoftShrink, Softmin, TanhShrink, Threshold, ThresholdedReLU,
};
pub use gelu::{fast_gelu, gelu, FastGELU, GELU};
pub use relu::{leaky_relu, relu, LeakyReLU, ReLU};
pub use sigmoid::{sigmoid, tanh, Sigmoid, Tanh};
pub use softmax::{log_softmax, softmax, LogSoftmax, Softmax, SoftmaxConfig};
pub use swish::{mish, silu, swish, ActivationKind, Mish, SiLU, Swish};

use brain_core::Tensor;

/// Trait for point-wise activation functions.
pub trait Activation: Send + Sync {
    /// Applies activation elementwise to input tensor.
    fn forward(&self, input: &Tensor) -> Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
