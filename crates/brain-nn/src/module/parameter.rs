//! # Trainable Parameters & Non-Trainable Buffers
//!
//! Parameter wrapper, non-trainable state buffers, and named parameter collections.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Trainable parameter wrapper holding tensor data and gradient eligibility.
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub tensor: Tensor,
    pub requires_grad: bool,
}

impl Parameter {
    pub fn new(name: impl Into<String>, tensor: Tensor) -> Self {
        Self {
            name: name.into(),
            tensor,
            requires_grad: true,
        }
    }
}

/// Non-trainable buffer (e.g. running mean/variance in BatchNorm).
#[derive(Debug, Clone)]
pub struct Buffer {
    pub name: String,
    pub tensor: Tensor,
}

impl Buffer {
    pub fn new(name: impl Into<String>, tensor: Tensor) -> Self {
        Self {
            name: name.into(),
            tensor,
        }
    }
}

/// Named parameter association.
#[derive(Debug, Clone)]
pub struct NamedParameter {
    pub name: String,
    pub tensor: Tensor,
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
