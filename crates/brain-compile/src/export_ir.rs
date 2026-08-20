//! # IR Exporters & Disassemblers
//!
//! Exports IR graphs to Graphviz Dot format, JSON AST, and human-readable text representations.

use crate::ir::IrGraph;

/// Exports graph to Graphviz DOT string format.
pub fn export_dot(graph: &IrGraph) -> String {
    let mut dot = String::from("digraph IrGraph {\n");
    for (i, node) in graph.nodes.iter().enumerate() {
        dot.push_str(&format!("  node_{} [label=\"{:?}\"];\n", i, node.kind));
    }
    dot.push_str("}\n");
    dot
}

/// Exports graph to human-readable disassembly text.
pub fn export_text(graph: &IrGraph) -> String {
    let mut text = String::from("IR Function @graph:\n");
    for node in &graph.nodes {
        text.push_str(&format!(
            "  %{} = {:?}({:?})\n",
            node.output, node.kind, node.inputs
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
