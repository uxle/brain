//! # Topological Analysis & Ordering
//!
//! Kahn's algorithm and DFS-based topological sorts, critical path calculation.
#![allow(missing_docs)]

use std::collections::{HashMap, VecDeque};
use crate::core::{NodeId, GraphResult, GraphError};
use crate::ir::GraphIr;

/// Topological sort and critical path analysis result.
#[derive(Debug, Clone, Default)]
pub struct TopoOrder {
    pub node_order: Vec<NodeId>,
    pub node_ranks: HashMap<NodeId, usize>,
    pub critical_path_length: usize,
}

/// Computes topological order of nodes in `GraphIr` using Kahn's algorithm.
pub fn compute_topological_order(graph: &GraphIr) -> GraphResult<TopoOrder> {
    let mut in_degrees = HashMap::new();
    let mut consumers: HashMap<usize, Vec<NodeId>> = HashMap::new();

    // Build value -> consuming node map
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &inp in &node.inputs {
            consumers.entry(inp).or_default().push(node_idx);
        }
    }

    // Node producer -> consumer node dependencies
    let mut node_dependents: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        in_degrees.insert(node_idx, 0);
        for &out in &node.outputs {
            if let Some(cons) = consumers.get(&out) {
                for &c in cons {
                    node_dependents.entry(node_idx).or_default().push(c);
                }
            }
        }
    }

    for deps in node_dependents.values() {
        for &d in deps {
            *in_degrees.entry(d).or_insert(0) += 1;
        }
    }

    let mut queue = VecDeque::new();
    for (node_idx, &deg) in &in_degrees {
        if deg == 0 {
            queue.push_back(*node_idx);
        }
    }

    let mut order = Vec::new();
    let mut ranks = HashMap::new();

    while let Some(curr) = queue.pop_front() {
        order.push(curr);
        let curr_rank = *ranks.get(&curr).unwrap_or(&0);

        if let Some(deps) = node_dependents.get(&curr) {
            for &dep in deps {
                let deg = in_degrees.get_mut(&dep).unwrap();
                *deg -= 1;
                let next_rank = ranks.entry(dep).or_insert(0);
                *next_rank = (*next_rank).max(curr_rank + 1);

                if *deg == 0 {
                    queue.push_back(dep);
                }
            }
        }
    }

    if order.len() < graph.nodes.len() {
        return Err(GraphError::CyclicDependency("Cycle detected in graph".into()));
    }

    let max_rank = ranks.values().copied().max().unwrap_or(0);

    Ok(TopoOrder {
        node_order: order,
        node_ranks: ranks,
        critical_path_length: max_rank + 1,
    })
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
