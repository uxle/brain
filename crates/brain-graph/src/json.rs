//! # JSON Serialization
//!
//! Pure Rust deterministic JSON serialization and deserialization of computation graphs.
#![allow(missing_docs)]

use crate::ir::GraphIr;

/// Serializes `GraphIr` to a formatted JSON string.
pub fn to_json(graph: &GraphIr) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"name\": \"{}\",\n", graph.name));
    out.push_str("  \"nodes\": [\n");

    for (i, node) in graph.nodes.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"id\": {},\n", node.id));
        out.push_str(&format!("      \"name\": \"{}\",\n", node.name));
        out.push_str(&format!("      \"op\": \"{}\",\n", node.op.name()));
        out.push_str(&format!("      \"inputs\": {:?},\n", node.inputs));
        out.push_str(&format!("      \"outputs\": {:?}\n", node.outputs));
        out.push_str("    }");
        if i + 1 < graph.nodes.len() { out.push(','); }
        out.push('\n');
    }

    out.push_str("  ],\n");
    out.push_str(&format!("  \"inputs\": {:?},\n", graph.inputs));
    out.push_str(&format!("  \"outputs\": {:?}\n", graph.outputs));
    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
