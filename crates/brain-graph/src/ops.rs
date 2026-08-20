//! # Graph Operator Constructors
//!
//! Direct helper functions to construct and apply graph operations.
#![allow(missing_docs)]

use crate::builder::GraphBuilder;
use crate::core::ValueId;
use crate::ir::ops::OpKind;
use brain_core::Tensor;

/// Adds an addition node to the builder.
pub fn graph_add(builder: &mut GraphBuilder, a: ValueId, b: ValueId, shape: Vec<usize>) -> ValueId {
    builder.add_node("add", OpKind::Add, vec![a, b], shape)
}

/// Adds a matrix multiplication node to the builder.
pub fn graph_matmul(
    builder: &mut GraphBuilder,
    a: ValueId,
    b: ValueId,
    shape: Vec<usize>,
) -> ValueId {
    builder.add_node("matmul", OpKind::MatMul, vec![a, b], shape)
}

/// Adds a ReLU activation node to the builder.
pub fn graph_relu(builder: &mut GraphBuilder, a: ValueId, shape: Vec<usize>) -> ValueId {
    builder.add_node("relu", OpKind::Relu, vec![a], shape)
}

/// Direct execution of OpKind on `brain_core::Tensor` inputs.
pub fn op_apply(op: OpKind, inputs: &[&Tensor]) -> Tensor {
    match op {
        OpKind::Add => {
            if inputs.len() >= 2 {
                inputs[0] + inputs[1]
            } else {
                Tensor::zeros(vec![1])
            }
        }
        OpKind::Sub => {
            if inputs.len() >= 2 {
                inputs[0] - inputs[1]
            } else {
                Tensor::zeros(vec![1])
            }
        }
        OpKind::Mul => {
            if inputs.len() >= 2 {
                inputs[0] * inputs[1]
            } else {
                Tensor::zeros(vec![1])
            }
        }
        OpKind::MatMul => {
            if inputs.len() >= 2
                && inputs[0].ndim() == 2
                && inputs[1].ndim() == 2
                && inputs[0].shape()[1] == inputs[1].shape()[0]
            {
                let a = inputs[0];
                let b = inputs[1];
                let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
                let (a_vec, b_vec) = (a.to_vec(), b.to_vec());
                let mut out = vec![0.0f64; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f64;
                        for p in 0..k {
                            sum += a_vec[i * k + p] * b_vec[p * n + j];
                        }
                        out[i * n + j] = sum;
                    }
                }
                Tensor::from_vec(out, vec![m, n])
            } else {
                Tensor::zeros(vec![1])
            }
        }
        OpKind::Relu => {
            if let Some(t) = inputs.first() {
                let data: Vec<f64> = t.to_vec().iter().map(|&v| v.max(0.0)).collect();
                Tensor::from_vec(data, t.shape().to_vec())
            } else {
                Tensor::zeros(vec![1])
            }
        }
        OpKind::Sigmoid => {
            if let Some(t) = inputs.first() {
                let data: Vec<f64> = t
                    .to_vec()
                    .iter()
                    .map(|&v| 1.0 / (1.0 + (-v).exp()))
                    .collect();
                Tensor::from_vec(data, t.shape().to_vec())
            } else {
                Tensor::zeros(vec![1])
            }
        }
        _ => Tensor::zeros(vec![1]),
    }
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
    use crate::builder::GraphBuilder;
    use crate::core::DType;
    use crate::ir::ops::OpKind;
    use brain_core::Tensor;
}
