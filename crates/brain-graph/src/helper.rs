//! # Demo Model Graph Builders
//!
//! Out-of-the-box constructors for standard network topologies (MLP, CNN, Transformer).
#![allow(missing_docs)]

use crate::core::DType;
use crate::builder::GraphBuilder;
use crate::ir::GraphIr;
use crate::ir::ops::OpKind;

/// Constructs an MLP computation graph with specified layer dimensions.
pub fn build_mlp_graph(in_dim: usize, hidden_dim: usize, out_dim: usize) -> GraphIr {
    let mut b = GraphBuilder::new("mlp");
    let x = b.add_input("x", vec![1, in_dim], DType::F32);
    let w1 = b.add_constant("w1", vec![in_dim, hidden_dim], vec![0.01; in_dim * hidden_dim]);
    let h1 = b.add_node("mm1", OpKind::MatMul, vec![x, w1], vec![1, hidden_dim]);
    let h1_act = b.add_node("relu1", OpKind::Relu, vec![h1], vec![1, hidden_dim]);

    let w2 = b.add_constant("w2", vec![hidden_dim, out_dim], vec![0.01; hidden_dim * out_dim]);
    let out = b.add_node("mm2", OpKind::MatMul, vec![h1_act, w2], vec![1, out_dim]);
    b.mark_output(out);

    b.build().unwrap()
}

/// Constructs a simple CNN computation graph.
pub fn build_cnn_graph(in_channels: usize, out_classes: usize) -> GraphIr {
    let mut b = GraphBuilder::new("cnn");
    let x = b.add_input("image", vec![1, in_channels, 28, 28], DType::F32);
    let w_conv = b.add_constant("conv_w", vec![16, in_channels, 3, 3], vec![0.01; 16 * in_channels * 9]);
    let conv1 = b.add_node("conv", OpKind::Conv2D, vec![x, w_conv], vec![1, 16, 26, 26]);
    let act1 = b.add_node("relu", OpKind::Relu, vec![conv1], vec![1, 16, 26, 26]);
    let flat = b.add_node("flatten", OpKind::Flatten, vec![act1], vec![1, 16 * 26 * 26]);

    let w_fc = b.add_constant("fc_w", vec![16 * 26 * 26, out_classes], vec![0.01; 16 * 26 * 26 * out_classes]);
    let logits = b.add_node("fc", OpKind::MatMul, vec![flat, w_fc], vec![1, out_classes]);
    b.mark_output(logits);

    b.build().unwrap()
}

/// Constructs a demo self-attention block graph.
pub fn build_transformer_graph(seq_len: usize, d_model: usize) -> GraphIr {
    let mut b = GraphBuilder::new("transformer_block");
    let x = b.add_input("tokens", vec![seq_len, d_model], DType::F32);
    let w_q = b.add_constant("w_q", vec![d_model, d_model], vec![0.01; d_model * d_model]);
    let q = b.add_node("q_proj", OpKind::MatMul, vec![x, w_q], vec![seq_len, d_model]);
    b.mark_output(q);

    b.build().unwrap()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
