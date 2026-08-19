//! # Graph Deep Cloning & Subgraph Extraction
//!
//! Deep copy of graphs with complete node and value identifier remapping.
#![allow(missing_docs)]

use std::collections::HashMap;
use crate::ir::GraphIr;

/// Clones a subgraph containing the specified node IDs.
pub fn clone_subgraph(graph: &GraphIr, node_ids: &[usize]) -> GraphIr {
    let mut new_graph = GraphIr::new(&format!("{}_subgraph", graph.name));
    let mut val_map = HashMap::new();

    for &nid in node_ids {
        if let Some(node) = graph.get_node(nid) {
            // Remap inputs
            let mut new_inputs = Vec::new();
            for &inp in &node.inputs {
                let mapped = *val_map.entry(inp).or_insert_with(|| {
                    let old_v = &graph.values[inp];
                    new_graph.add_value(&old_v.name, old_v.shape.clone(), old_v.dtype)
                });
                new_inputs.push(mapped);
            }

            // Remap outputs
            let mut new_outputs = Vec::new();
            for &out in &node.outputs {
                let mapped = *val_map.entry(out).or_insert_with(|| {
                    let old_v = &graph.values[out];
                    new_graph.add_value(&old_v.name, old_v.shape.clone(), old_v.dtype)
                });
                new_outputs.push(mapped);
            }

            new_graph.add_node(&node.name, node.op, new_inputs, new_outputs);
        }
    }

    new_graph
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
