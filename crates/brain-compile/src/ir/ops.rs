//! # Complete IR Operation Catalog
//!
//! Exhaustive catalog of tensor operations, elementwise math, reductions, and fused kernels.

/// Operation kinds supported by the IR engine.
#[derive(Debug, Clone, PartialEq)]
pub enum OpKind {
    Constant(f64),
    Parameter(String),
    Input(usize),
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Exp,
    Log,
    Sin,
    Cos,
    Relu,
    Sigmoid,
    Tanh,
    MatMul,
    Conv2d { stride: usize, padding: usize },
    MaxPool2d { kernel_size: usize },
    ReduceSum { dim: Option<usize> },
    ReduceMean { dim: Option<usize> },
    Reshape(Vec<usize>),
    Transpose(Vec<usize>),
    Broadcast(Vec<usize>),
    FusedElementwise(Vec<String>),
}

impl OpKind {
    /// Returns whether this operation is purely elementwise.
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Sub
                | Self::Mul
                | Self::Div
                | Self::Neg
                | Self::Exp
                | Self::Log
                | Self::Sin
                | Self::Cos
                | Self::Relu
                | Self::Sigmoid
                | Self::Tanh
                | Self::FusedElementwise(_)
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
