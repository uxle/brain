//! # ONNX Graph Evaluation Engine
//!
//! Direct interpretive evaluation of imported ONNX graphs using pure Rust kernel implementations.
#![allow(missing_docs)]

pub mod checker;
pub use checker::{check_model, CheckerReport};

use crate::config::EvalConfig;
use crate::core::{OnnxError, OnnxResult};
use crate::ir::OnnxModel;
use brain_core::Tensor;
use std::collections::HashMap;

/// Evaluates an ONNX model given input tensors, returning output tensors.
pub fn evaluate_onnx_model(
    model: &OnnxModel,
    inputs: &HashMap<String, Tensor>,
    _config: &EvalConfig,
) -> OnnxResult<HashMap<String, Tensor>> {
    let mut env: HashMap<String, Tensor> = inputs.clone();

    // Load initializers
    for (name, val) in &model.graph.values {
        if val.is_initializer {
            if let Some(ref t) = val.tensor_data {
                env.insert(name.clone(), t.clone());
            }
        }
    }

    // Step through nodes topologically
    for node in &model.graph.nodes {
        match node.op_type.as_str() {
            "Relu" => {
                let in_t = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                let out_vec: Vec<f64> = in_t.to_vec().iter().map(|&x| x.max(0.0)).collect();
                let out_t = Tensor::from_vec(out_vec, in_t.shape().to_vec());
                env.insert(node.outputs[0].clone(), out_t);
            }
            "Add" => {
                let t1 = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                let t2 = env.get(&node.inputs[1]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[1]))
                })?;
                env.insert(node.outputs[0].clone(), t1 + t2);
            }
            "Sub" => {
                let t1 = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                let t2 = env.get(&node.inputs[1]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[1]))
                })?;
                env.insert(node.outputs[0].clone(), t1 - t2);
            }
            "Mul" => {
                let t1 = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                let t2 = env.get(&node.inputs[1]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[1]))
                })?;
                env.insert(node.outputs[0].clone(), t1 * t2);
            }
            "Div" => {
                let t1 = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                let t2 = env.get(&node.inputs[1]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[1]))
                })?;
                let v1 = t1.to_vec();
                let v2 = t2.to_vec();
                let out: Vec<f64> = v1.iter().zip(v2.iter()).map(|(&a, &b)| a / b).collect();
                env.insert(
                    node.outputs[0].clone(),
                    Tensor::from_vec(out, t1.shape().to_vec()),
                );
            }
            "MatMul" => {
                let t1 = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                let t2 = env.get(&node.inputs[1]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[1]))
                })?;
                if t1.ndim() != 2 || t2.ndim() != 2 || t1.shape()[1] != t2.shape()[0] {
                    return Err(OnnxError::InvalidTensorShape(format!(
                        "MatMul shape mismatch: {:?} vs {:?}",
                        t1.shape(),
                        t2.shape()
                    )));
                }
                let (m, k, n) = (t1.shape()[0], t1.shape()[1], t2.shape()[1]);
                let (d1, d2) = (t1.to_vec(), t2.to_vec());
                let mut out = vec![0.0f64; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f64;
                        for p in 0..k {
                            sum += d1[i * k + p] * d2[p * n + j];
                        }
                        out[i * n + j] = sum;
                    }
                }
                env.insert(node.outputs[0].clone(), Tensor::from_vec(out, vec![m, n]));
            }
            "Gemm" => {
                let a = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                let b = env.get(&node.inputs[1]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[1]))
                })?;
                let c_opt = node.inputs.get(2).and_then(|name| env.get(name));

                let alpha: f64 = node
                    .attributes
                    .get("alpha")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0);
                let beta: f64 = node
                    .attributes
                    .get("beta")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0);
                let trans_a = node
                    .attributes
                    .get("transA")
                    .map(|s| s == "1" || s == "true")
                    .unwrap_or(false);
                let trans_b = node
                    .attributes
                    .get("transB")
                    .map(|s| s == "1" || s == "true")
                    .unwrap_or(false);

                let a_mat = if trans_a {
                    a.transpose(0, 1)
                } else {
                    a.clone()
                };
                let b_mat = if trans_b {
                    b.transpose(0, 1)
                } else {
                    b.clone()
                };

                if a_mat.ndim() != 2 || b_mat.ndim() != 2 || a_mat.shape()[1] != b_mat.shape()[0] {
                    return Err(OnnxError::InvalidTensorShape(format!(
                        "Gemm shape mismatch: {:?} vs {:?}",
                        a_mat.shape(),
                        b_mat.shape()
                    )));
                }

                let (m, k, n) = (a_mat.shape()[0], a_mat.shape()[1], b_mat.shape()[1]);
                let (d1, d2) = (a_mat.to_vec(), b_mat.to_vec());
                let mut out = vec![0.0f64; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f64;
                        for p in 0..k {
                            sum += d1[i * k + p] * d2[p * n + j];
                        }
                        let mut val = alpha * sum;
                        if let Some(c) = c_opt {
                            let c_vec = c.to_vec();
                            let c_val = if c_vec.len() == 1 {
                                c_vec[0]
                            } else if c_vec.len() == n {
                                c_vec[j]
                            } else if c_vec.len() == m * n {
                                c_vec[i * n + j]
                            } else {
                                0.0
                            };
                            val += beta * c_val;
                        }
                        out[i * n + j] = val;
                    }
                }
                env.insert(node.outputs[0].clone(), Tensor::from_vec(out, vec![m, n]));
            }
            "Sigmoid" => {
                let in_t = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                let out_vec: Vec<f64> = in_t
                    .to_vec()
                    .iter()
                    .map(|&x| 1.0 / (1.0 + (-x).exp()))
                    .collect();
                env.insert(
                    node.outputs[0].clone(),
                    Tensor::from_vec(out_vec, in_t.shape().to_vec()),
                );
            }
            "Tanh" => {
                let in_t = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                let out_vec: Vec<f64> = in_t.to_vec().iter().map(|&x| x.tanh()).collect();
                env.insert(
                    node.outputs[0].clone(),
                    Tensor::from_vec(out_vec, in_t.shape().to_vec()),
                );
            }
            "Identity" => {
                let in_t = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                env.insert(node.outputs[0].clone(), in_t.clone());
            }
            "QuantizeLinear" => {
                let in_t = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                let scale = env
                    .get(&node.inputs[1])
                    .map(|t| t.to_vec().first().copied().unwrap_or(1.0))
                    .unwrap_or(1.0);
                let zp = node
                    .inputs
                    .get(2)
                    .and_then(|name| env.get(name))
                    .map(|t| t.to_vec().first().copied().unwrap_or(0.0))
                    .unwrap_or(0.0);
                let out_vec: Vec<f64> = in_t
                    .to_vec()
                    .iter()
                    .map(|&x| ((x / scale).round() + zp).clamp(-128.0, 127.0))
                    .collect();
                env.insert(
                    node.outputs[0].clone(),
                    Tensor::from_vec(out_vec, in_t.shape().to_vec()),
                );
            }
            "DequantizeLinear" => {
                let in_t = env.get(&node.inputs[0]).ok_or_else(|| {
                    OnnxError::GraphLoweringError(format!("Missing tensor {}", node.inputs[0]))
                })?;
                let scale = env
                    .get(&node.inputs[1])
                    .map(|t| t.to_vec().first().copied().unwrap_or(1.0))
                    .unwrap_or(1.0);
                let zp = node
                    .inputs
                    .get(2)
                    .and_then(|name| env.get(name))
                    .map(|t| t.to_vec().first().copied().unwrap_or(0.0))
                    .unwrap_or(0.0);
                let out_vec: Vec<f64> = in_t.to_vec().iter().map(|&x| (x - zp) * scale).collect();
                env.insert(
                    node.outputs[0].clone(),
                    Tensor::from_vec(out_vec, in_t.shape().to_vec()),
                );
            }
            _ => {
                return Err(OnnxError::UnsupportedOp {
                    op_type: node.op_type.clone(),
                    domain: node.domain.clone(),
                });
            }
        }
    }

    let mut results = HashMap::new();
    for out_name in &model.graph.outputs {
        if let Some(t) = env.get(out_name) {
            results.insert(out_name.clone(), t.clone());
        }
    }

    Ok(results)
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
