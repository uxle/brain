//! # Activation Layer Modules
//!
//! Object-oriented `Module` wrappers for point-wise activation functions.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};
use crate::activations::{relu, sigmoid, tanh, gelu, silu, mish};

macro_rules! impl_activation_module {
    ($name:ident, $func:ident) => {
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;
        impl Module for $name {
            fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
                Ok($func(input))
            }
        }
    };
}

impl_activation_module!(ReLU, relu);
impl_activation_module!(Sigmoid, sigmoid);
impl_activation_module!(Tanh, tanh);
impl_activation_module!(GELU, gelu);
impl_activation_module!(SiLU, silu);
impl_activation_module!(Mish, mish);

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
