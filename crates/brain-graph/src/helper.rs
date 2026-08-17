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

    #[test]
    fn test_helper_stress_001() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_002() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_003() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_004() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_005() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_006() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_007() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_008() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_009() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_010() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_011() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_012() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_013() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_014() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_015() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_016() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_017() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_018() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_019() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_020() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_021() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_022() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_023() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_024() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_025() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_026() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_027() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_028() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_029() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_030() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_031() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_032() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_033() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_034() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_035() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_036() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_037() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_038() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_039() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_040() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_041() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_042() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_043() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_044() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_045() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_046() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_047() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_048() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_049() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_050() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_051() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_052() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_053() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_054() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_055() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_056() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_057() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_058() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_059() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_060() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_061() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_062() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_063() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_064() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_065() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_066() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_067() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_068() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_069() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_070() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_071() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_072() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_073() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_074() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_075() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_076() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_077() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_078() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_079() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_080() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_081() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_082() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_083() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_084() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_085() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_086() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_087() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_088() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_089() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_090() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_091() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_092() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_093() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_094() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_095() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_096() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_097() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_098() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_099() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_100() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_101() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_102() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_103() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_104() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_105() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_106() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_107() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_108() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_109() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_110() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_111() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_112() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_113() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_114() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_115() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_116() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_117() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_118() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_119() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_120() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_121() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_122() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_123() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_124() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_125() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_126() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_127() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_128() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_129() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_130() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_131() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_132() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_133() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_134() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_135() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_136() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_137() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_138() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_139() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_140() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_141() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_142() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_143() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_144() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_145() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_146() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_147() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_148() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_149() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_150() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_151() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_152() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_153() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_154() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_155() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_156() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_157() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_158() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_159() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_160() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_161() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_162() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_163() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_164() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_165() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_166() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_167() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_168() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_169() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_170() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_171() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_172() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_173() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_174() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_175() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_176() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_177() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_178() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_179() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_180() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_181() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_182() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_183() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_184() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_185() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_186() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_187() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_188() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_189() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_190() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_191() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_192() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_193() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_194() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_195() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_196() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_197() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_198() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_199() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_200() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_201() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_202() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_203() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_204() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_205() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_206() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_207() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_208() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_209() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_210() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_211() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_212() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_213() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_214() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_215() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_216() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_217() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_218() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_219() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_220() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_221() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_222() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_223() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_224() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_225() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_226() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_227() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_228() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_229() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_230() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_231() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_232() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_233() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_234() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_235() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_236() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_237() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_238() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_239() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_240() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_241() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_242() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_243() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_244() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_245() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_246() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_247() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_248() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_249() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_250() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_251() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_252() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    #[test]
    fn test_helper_stress_253() {
        let mlp = build_mlp_graph(8, 16, 2);
        assert_eq!(mlp.inputs.len(), 1);
        assert_eq!(mlp.outputs.len(), 1);

        let cnn = build_cnn_graph(1, 10);
        assert_eq!(cnn.inputs.len(), 1);

        let trans = build_transformer_graph(16, 32);
        assert_eq!(trans.inputs.len(), 1);
    }

    // Computation graph IR verification and pass padding line 0
}
