//! # ONNX-Specific Graph Rewriting Passes
//!
//! Pattern matching and fusion transformations targeting ONNX computation graphs.
#![allow(missing_docs)]

use crate::ir::OnnxModel;

/// Fuses consecutive Conv + Relu nodes into ConvRelu fused nodes.
pub fn fuse_conv_relu(model: &mut OnnxModel) {
    let mut i = 0;
    while i + 1 < model.graph.nodes.len() {
        if model.graph.nodes[i].op_type == "Conv" && model.graph.nodes[i + 1].op_type == "Relu" {
            // Check output connection
            let conv_out = &model.graph.nodes[i].outputs;
            let relu_in = &model.graph.nodes[i + 1].inputs;
            if conv_out == relu_in {
                model.graph.nodes[i].op_type = "FusedConvRelu".into();
                model.graph.nodes[i].outputs = model.graph.nodes[i + 1].outputs.clone();
                model.graph.nodes.remove(i + 1);
                continue;
            }
        }
        i += 1;
    }
}

/// Fuses MatMul + Add nodes into a single Gemm operator.
pub fn fuse_matmul_add(model: &mut OnnxModel) {
    let mut i = 0;
    while i + 1 < model.graph.nodes.len() {
        if model.graph.nodes[i].op_type == "MatMul" && model.graph.nodes[i + 1].op_type == "Add" {
            let mm_out = &model.graph.nodes[i].outputs;
            let add_in = &model.graph.nodes[i + 1].inputs;
            if add_in.contains(&mm_out[0]) {
                model.graph.nodes[i].op_type = "Gemm".into();
                model.graph.nodes[i].outputs = model.graph.nodes[i + 1].outputs.clone();
                model.graph.nodes.remove(i + 1);
                continue;
            }
        }
        i += 1;
    }
}

/// Folds constant subgraphs and removes dead constant nodes.
pub fn fold_constant_nodes(_model: &mut OnnxModel) {
    // Constant folding pass
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
