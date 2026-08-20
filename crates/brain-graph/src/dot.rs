//! # Graphviz DOT Exporter
//!
//! Visualizes computation graph IRs in Graphviz DOT format with color-coded nodes.
#![allow(missing_docs)]

use crate::ir::GraphIr;

/// Converts a `GraphIr` into a Graphviz DOT representation.
pub fn to_dot(graph: &GraphIr) -> String {
    let mut dot = String::new();
    dot.push_str(&format!("digraph \"{}\" {{\n", graph.name));
    dot.push_str("    node [shape=box, style=\"rounded,filled\", fillcolor=\"#f0f0f0\", fontname=\"Helvetica\"];\n");
    dot.push_str("    edge [fontname=\"Helvetica\", fontsize=10];\n\n");

    // Add inputs
    for &inp in &graph.inputs {
        let v = &graph.values[inp];
        dot.push_str(&format!(
            "    val_{} [shape=ellipse, fillcolor=\"#cce5ff\", label=\"{} {:?}\"];\n",
            inp, v.name, v.shape.dims
        ));
    }

    // Add nodes
    for node in &graph.nodes {
        let color = match node.op {
            crate::ir::ops::OpKind::MatMul | crate::ir::ops::OpKind::Conv2D => "#d4edda",
            crate::ir::ops::OpKind::Relu | crate::ir::ops::OpKind::Sigmoid => "#fff3cd",
            _ => "#e2e3e5",
        };
        dot.push_str(&format!(
            "    node_{} [label=\"{} ({:?})\", fillcolor=\"{}\"];\n",
            node.id, node.name, node.op, color
        ));

        for &inp in &node.inputs {
            dot.push_str(&format!("    val_{} -> node_{};\n", inp, node.id));
        }
        for &out in &node.outputs {
            let v = &graph.values[out];
            dot.push_str(&format!(
                "    val_{} [shape=ellipse, label=\"{} {:?}\"];\n",
                out, v.name, v.shape.dims
            ));
            dot.push_str(&format!("    node_{} -> val_{};\n", node.id, out));
        }
    }

    dot.push_str("}\n");
    dot
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
