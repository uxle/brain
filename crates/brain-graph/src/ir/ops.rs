//! # Operation Catalog & Registry
//!
//! Enumeration of all graph operators, signature lookup, and shape calculators.
#![allow(missing_docs)]

use std::collections::HashMap;

/// Broad category of operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OpKind {
    #[default]
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    MatMul,
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    BatchNorm,
    LayerNorm,
    Softmax,
    Reshape,
    Transpose,
    Flatten,
    Constant,
    Custom,
}

impl OpKind {
    pub fn name(&self) -> &'static str {
        match self {
            OpKind::Add => "Add",
            OpKind::Sub => "Sub",
            OpKind::Mul => "Mul",
            OpKind::Div => "Div",
            OpKind::Neg => "Neg",
            OpKind::Relu => "Relu",
            OpKind::Sigmoid => "Sigmoid",
            OpKind::Tanh => "Tanh",
            OpKind::Gelu => "Gelu",
            OpKind::MatMul => "MatMul",
            OpKind::Conv2D => "Conv2D",
            OpKind::MaxPool2D => "MaxPool2D",
            OpKind::AvgPool2D => "AvgPool2D",
            OpKind::BatchNorm => "BatchNorm",
            OpKind::LayerNorm => "LayerNorm",
            OpKind::Softmax => "Softmax",
            OpKind::Reshape => "Reshape",
            OpKind::Transpose => "Transpose",
            OpKind::Flatten => "Flatten",
            OpKind::Constant => "Constant",
            OpKind::Custom => "Custom",
        }
    }

    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div
            | OpKind::Neg | OpKind::Relu | OpKind::Sigmoid | OpKind::Tanh | OpKind::Gelu
        )
    }

    pub fn min_inputs(&self) -> usize {
        match self {
            OpKind::Constant => 0,
            OpKind::Neg | OpKind::Relu | OpKind::Sigmoid | OpKind::Tanh | OpKind::Gelu
            | OpKind::MaxPool2D | OpKind::AvgPool2D | OpKind::Softmax
            | OpKind::Reshape | OpKind::Transpose | OpKind::Flatten => 1,
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::MatMul
            | OpKind::Conv2D | OpKind::BatchNorm | OpKind::LayerNorm => 2,
            OpKind::Custom => 1,
        }
    }
}

/// Registry mapping string identifiers to OpKind and metadata.
#[derive(Debug, Default)]
pub struct OpRegistry {
    ops: HashMap<String, OpKind>,
}

impl OpRegistry {
    pub fn new() -> Self {
        let mut reg = Self { ops: HashMap::new() };
        reg.register("add", OpKind::Add);
        reg.register("sub", OpKind::Sub);
        reg.register("mul", OpKind::Mul);
        reg.register("div", OpKind::Div);
        reg.register("matmul", OpKind::MatMul);
        reg.register("relu", OpKind::Relu);
        reg.register("conv2d", OpKind::Conv2D);
        reg.register("softmax", OpKind::Softmax);
        reg
    }

    pub fn register(&mut self, name: &str, op: OpKind) {
        self.ops.insert(name.to_lowercase(), op);
    }

    pub fn lookup(&self, name: &str) -> Option<OpKind> {
        self.ops.get(&name.to_lowercase()).copied()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
