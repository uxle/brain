//! # Topological Graph Traversal and Cycle Detection
//!
//! Provides topological sorting of computational DAG nodes with cycle detection
//! and unreachable path pruning.

use crate::value::Value;
use brain_core::{BrainError, BrainResult};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

enum Frame {
    Visit(Arc<Value>),
    Finalize(Arc<Value>),
}

/// Computes a reverse-topological order of nodes starting from a root value.
///
/// Implemented as an **iterative** post-order DFS using an explicit stack so
/// that deeply unrolled computation graphs (e.g. thousands of chained ops)
/// cannot overflow the program stack. Cycle detection is preserved: a node
/// encountered while still "on the active path" (gray) reports a cycle.
pub fn topological_sort(root: &Value) -> BrainResult<Vec<Arc<Value>>> {
    let mut order: Vec<Arc<Value>> = Vec::new();
    let mut done: HashSet<usize> = HashSet::new(); // fully processed (black)
    let mut active: HashSet<usize> = HashSet::new(); // on current path (gray)

    let root_arc = Arc::new(root.clone());
    let mut stack = vec![Frame::Visit(root_arc)];

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Finalize(node) => {
                active.remove(&node.id());
                done.insert(node.id());
                order.push(node);
            }
            Frame::Visit(node) => {
                let id = node.id();
                if active.contains(&id) {
                    return Err(BrainError::invalid_value(format!(
                        "Cycle detected in computation graph at node ID {}",
                        id
                    )));
                }
                if done.contains(&id) {
                    continue;
                }
                active.insert(id);
                stack.push(Frame::Finalize(Arc::clone(&node)));
                for parent in node.grad_fn().parents() {
                    stack.push(Frame::Visit(Arc::clone(parent)));
                }
            }
        }
    }

    Ok(order)
}

/// Computes execution levels for (optionally) parallel topological backward
/// evaluation. Nodes are bucketed by their longest dependency chain from a
/// leaf: level 0 = leaves, level k = 1 + max(level of parents). Within a
/// level every node's parents live in strictly earlier levels, so levels may
/// be processed concurrently.
pub fn compute_dag_levels(ordered_nodes: &[Arc<Value>]) -> Vec<Vec<Arc<Value>>> {
    if ordered_nodes.is_empty() {
        return Vec::new();
    }

    let mut level_of: HashMap<usize, usize> = HashMap::new();
    let mut buckets: Vec<Vec<Arc<Value>>> = Vec::new();

    for node in ordered_nodes {
        let parents = node.grad_fn().parents();
        let lvl = if parents.is_empty() {
            0
        } else {
            parents
                .iter()
                .map(|p| *level_of.get(&p.id()).unwrap_or(&0))
                .max()
                .map(|m| m + 1)
                .unwrap_or(1)
        };
        level_of.insert(node.id(), lvl);
        while buckets.len() <= lvl {
            buckets.push(Vec::new());
        }
        buckets[lvl].push(Arc::clone(node));
    }

    buckets
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;
}
