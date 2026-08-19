//! # Autograd Forward Operations Subsystem
//!
//! Exposes differentiable forward operator primitives and gradient rules.

pub mod activation_grad;
pub mod binary;
pub mod broadcast_grad;
pub mod conv_grad;
pub mod fft_grad;
pub mod index_grad;
pub mod linalg_grad;
pub mod pool_grad;
pub mod quant_grad;
pub mod reduction_grad;
pub mod sparse_grad;
pub mod tensor_grad;
pub mod unary;

pub use binary::{add, div, matmul, max_elem, min_elem, mul, pow, sub, where_cond};
pub use conv_grad::{conv2d, conv_transpose2d};
pub use pool_grad::{avg_pool2d, max_pool2d};
pub use unary::{
    abs, clamp, cos, exp, log, log_softmax, mean, neg, recip, relu, sign, sigmoid, sin, softmax,
    sqrt, square, sum, tanh,
};
pub use activation_grad::{grad_gelu, grad_leaky_relu, grad_silu};
