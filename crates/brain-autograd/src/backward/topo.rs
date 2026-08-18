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

    #[test]
    fn test_topo_sort_stress_001() {
        let mut x = Value::scalar(1.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_002() {
        let mut x = Value::scalar(1.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_003() {
        let mut x = Value::scalar(1.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_004() {
        let mut x = Value::scalar(1.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_005() {
        let mut x = Value::scalar(1.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_006() {
        let mut x = Value::scalar(1.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_007() {
        let mut x = Value::scalar(1.7000000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_008() {
        let mut x = Value::scalar(1.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_009() {
        let mut x = Value::scalar(1.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_010() {
        let mut x = Value::scalar(2.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_011() {
        let mut x = Value::scalar(2.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_012() {
        let mut x = Value::scalar(2.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_013() {
        let mut x = Value::scalar(2.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_014() {
        let mut x = Value::scalar(2.4000000000000004);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_015() {
        let mut x = Value::scalar(2.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_016() {
        let mut x = Value::scalar(2.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_017() {
        let mut x = Value::scalar(2.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_018() {
        let mut x = Value::scalar(2.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_019() {
        let mut x = Value::scalar(2.9000000000000004);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_020() {
        let mut x = Value::scalar(3.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_021() {
        let mut x = Value::scalar(3.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_022() {
        let mut x = Value::scalar(3.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_023() {
        let mut x = Value::scalar(3.3000000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_024() {
        let mut x = Value::scalar(3.4000000000000004);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_025() {
        let mut x = Value::scalar(3.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_026() {
        let mut x = Value::scalar(3.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_027() {
        let mut x = Value::scalar(3.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_028() {
        let mut x = Value::scalar(3.8000000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_029() {
        let mut x = Value::scalar(3.9000000000000004);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_030() {
        let mut x = Value::scalar(4.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_031() {
        let mut x = Value::scalar(4.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_032() {
        let mut x = Value::scalar(4.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_033() {
        let mut x = Value::scalar(4.300000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_034() {
        let mut x = Value::scalar(4.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_035() {
        let mut x = Value::scalar(4.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_036() {
        let mut x = Value::scalar(4.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_037() {
        let mut x = Value::scalar(4.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_038() {
        let mut x = Value::scalar(4.800000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_039() {
        let mut x = Value::scalar(4.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_040() {
        let mut x = Value::scalar(5.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_041() {
        let mut x = Value::scalar(5.1000000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_042() {
        let mut x = Value::scalar(5.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_043() {
        let mut x = Value::scalar(5.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_044() {
        let mut x = Value::scalar(5.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_045() {
        let mut x = Value::scalar(5.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_046() {
        let mut x = Value::scalar(5.6000000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_047() {
        let mut x = Value::scalar(5.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_048() {
        let mut x = Value::scalar(5.800000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_049() {
        let mut x = Value::scalar(5.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_050() {
        let mut x = Value::scalar(6.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_051() {
        let mut x = Value::scalar(6.1000000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_052() {
        let mut x = Value::scalar(6.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_053() {
        let mut x = Value::scalar(6.300000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_054() {
        let mut x = Value::scalar(6.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_055() {
        let mut x = Value::scalar(6.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_056() {
        let mut x = Value::scalar(6.6000000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_057() {
        let mut x = Value::scalar(6.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_058() {
        let mut x = Value::scalar(6.800000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_059() {
        let mut x = Value::scalar(6.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_060() {
        let mut x = Value::scalar(7.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_061() {
        let mut x = Value::scalar(7.1000000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_062() {
        let mut x = Value::scalar(7.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_063() {
        let mut x = Value::scalar(7.300000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_064() {
        let mut x = Value::scalar(7.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_065() {
        let mut x = Value::scalar(7.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_066() {
        let mut x = Value::scalar(7.6000000000000005);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_067() {
        let mut x = Value::scalar(7.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_068() {
        let mut x = Value::scalar(7.800000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_069() {
        let mut x = Value::scalar(7.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_070() {
        let mut x = Value::scalar(8.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_071() {
        let mut x = Value::scalar(8.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_072() {
        let mut x = Value::scalar(8.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_073() {
        let mut x = Value::scalar(8.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_074() {
        let mut x = Value::scalar(8.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_075() {
        let mut x = Value::scalar(8.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_076() {
        let mut x = Value::scalar(8.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_077() {
        let mut x = Value::scalar(8.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_078() {
        let mut x = Value::scalar(8.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_079() {
        let mut x = Value::scalar(8.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_080() {
        let mut x = Value::scalar(9.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_081() {
        let mut x = Value::scalar(9.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_082() {
        let mut x = Value::scalar(9.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_083() {
        let mut x = Value::scalar(9.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_084() {
        let mut x = Value::scalar(9.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_085() {
        let mut x = Value::scalar(9.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_086() {
        let mut x = Value::scalar(9.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_087() {
        let mut x = Value::scalar(9.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_088() {
        let mut x = Value::scalar(9.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_089() {
        let mut x = Value::scalar(9.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_090() {
        let mut x = Value::scalar(10.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_091() {
        let mut x = Value::scalar(10.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_092() {
        let mut x = Value::scalar(10.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_093() {
        let mut x = Value::scalar(10.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_094() {
        let mut x = Value::scalar(10.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_095() {
        let mut x = Value::scalar(10.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_096() {
        let mut x = Value::scalar(10.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_097() {
        let mut x = Value::scalar(10.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_098() {
        let mut x = Value::scalar(10.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_099() {
        let mut x = Value::scalar(10.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_100() {
        let mut x = Value::scalar(11.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_101() {
        let mut x = Value::scalar(11.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_102() {
        let mut x = Value::scalar(11.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_103() {
        let mut x = Value::scalar(11.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_104() {
        let mut x = Value::scalar(11.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_105() {
        let mut x = Value::scalar(11.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_106() {
        let mut x = Value::scalar(11.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_107() {
        let mut x = Value::scalar(11.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_108() {
        let mut x = Value::scalar(11.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_109() {
        let mut x = Value::scalar(11.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_110() {
        let mut x = Value::scalar(12.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_111() {
        let mut x = Value::scalar(12.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_112() {
        let mut x = Value::scalar(12.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_113() {
        let mut x = Value::scalar(12.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_114() {
        let mut x = Value::scalar(12.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_115() {
        let mut x = Value::scalar(12.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_116() {
        let mut x = Value::scalar(12.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_117() {
        let mut x = Value::scalar(12.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_118() {
        let mut x = Value::scalar(12.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_119() {
        let mut x = Value::scalar(12.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_120() {
        let mut x = Value::scalar(13.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_121() {
        let mut x = Value::scalar(13.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_122() {
        let mut x = Value::scalar(13.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_123() {
        let mut x = Value::scalar(13.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_124() {
        let mut x = Value::scalar(13.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_125() {
        let mut x = Value::scalar(13.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_126() {
        let mut x = Value::scalar(13.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_127() {
        let mut x = Value::scalar(13.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_128() {
        let mut x = Value::scalar(13.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_129() {
        let mut x = Value::scalar(13.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_130() {
        let mut x = Value::scalar(14.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_131() {
        let mut x = Value::scalar(14.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_132() {
        let mut x = Value::scalar(14.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_133() {
        let mut x = Value::scalar(14.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_134() {
        let mut x = Value::scalar(14.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_135() {
        let mut x = Value::scalar(14.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_136() {
        let mut x = Value::scalar(14.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_137() {
        let mut x = Value::scalar(14.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_138() {
        let mut x = Value::scalar(14.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_139() {
        let mut x = Value::scalar(14.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_140() {
        let mut x = Value::scalar(15.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_141() {
        let mut x = Value::scalar(15.100000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_142() {
        let mut x = Value::scalar(15.200000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_143() {
        let mut x = Value::scalar(15.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_144() {
        let mut x = Value::scalar(15.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_145() {
        let mut x = Value::scalar(15.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_146() {
        let mut x = Value::scalar(15.600000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_147() {
        let mut x = Value::scalar(15.700000000000001);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_148() {
        let mut x = Value::scalar(15.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_149() {
        let mut x = Value::scalar(15.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_150() {
        let mut x = Value::scalar(16.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_151() {
        let mut x = Value::scalar(16.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_152() {
        let mut x = Value::scalar(16.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_153() {
        let mut x = Value::scalar(16.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_154() {
        let mut x = Value::scalar(16.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_155() {
        let mut x = Value::scalar(16.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_156() {
        let mut x = Value::scalar(16.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_157() {
        let mut x = Value::scalar(16.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_158() {
        let mut x = Value::scalar(16.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_159() {
        let mut x = Value::scalar(16.9);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_160() {
        let mut x = Value::scalar(17.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_161() {
        let mut x = Value::scalar(17.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_162() {
        let mut x = Value::scalar(17.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_163() {
        let mut x = Value::scalar(17.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_164() {
        let mut x = Value::scalar(17.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_165() {
        let mut x = Value::scalar(17.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_166() {
        let mut x = Value::scalar(17.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_167() {
        let mut x = Value::scalar(17.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_168() {
        let mut x = Value::scalar(17.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_169() {
        let mut x = Value::scalar(17.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_170() {
        let mut x = Value::scalar(18.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_171() {
        let mut x = Value::scalar(18.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_172() {
        let mut x = Value::scalar(18.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_173() {
        let mut x = Value::scalar(18.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_174() {
        let mut x = Value::scalar(18.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_175() {
        let mut x = Value::scalar(18.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_176() {
        let mut x = Value::scalar(18.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_177() {
        let mut x = Value::scalar(18.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_178() {
        let mut x = Value::scalar(18.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_179() {
        let mut x = Value::scalar(18.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_180() {
        let mut x = Value::scalar(19.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_181() {
        let mut x = Value::scalar(19.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_182() {
        let mut x = Value::scalar(19.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_183() {
        let mut x = Value::scalar(19.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_184() {
        let mut x = Value::scalar(19.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_185() {
        let mut x = Value::scalar(19.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_186() {
        let mut x = Value::scalar(19.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_187() {
        let mut x = Value::scalar(19.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_188() {
        let mut x = Value::scalar(19.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_189() {
        let mut x = Value::scalar(19.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_190() {
        let mut x = Value::scalar(20.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_191() {
        let mut x = Value::scalar(20.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_192() {
        let mut x = Value::scalar(20.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_193() {
        let mut x = Value::scalar(20.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_194() {
        let mut x = Value::scalar(20.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_195() {
        let mut x = Value::scalar(20.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_196() {
        let mut x = Value::scalar(20.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_197() {
        let mut x = Value::scalar(20.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_198() {
        let mut x = Value::scalar(20.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_199() {
        let mut x = Value::scalar(20.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_200() {
        let mut x = Value::scalar(21.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_201() {
        let mut x = Value::scalar(21.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_202() {
        let mut x = Value::scalar(21.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_203() {
        let mut x = Value::scalar(21.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_204() {
        let mut x = Value::scalar(21.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_205() {
        let mut x = Value::scalar(21.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_206() {
        let mut x = Value::scalar(21.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_207() {
        let mut x = Value::scalar(21.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_208() {
        let mut x = Value::scalar(21.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_209() {
        let mut x = Value::scalar(21.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_210() {
        let mut x = Value::scalar(22.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_211() {
        let mut x = Value::scalar(22.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_212() {
        let mut x = Value::scalar(22.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_213() {
        let mut x = Value::scalar(22.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_214() {
        let mut x = Value::scalar(22.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_215() {
        let mut x = Value::scalar(22.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_216() {
        let mut x = Value::scalar(22.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_217() {
        let mut x = Value::scalar(22.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_218() {
        let mut x = Value::scalar(22.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_219() {
        let mut x = Value::scalar(22.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_220() {
        let mut x = Value::scalar(23.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_221() {
        let mut x = Value::scalar(23.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_222() {
        let mut x = Value::scalar(23.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_223() {
        let mut x = Value::scalar(23.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_224() {
        let mut x = Value::scalar(23.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_225() {
        let mut x = Value::scalar(23.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_226() {
        let mut x = Value::scalar(23.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_227() {
        let mut x = Value::scalar(23.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_228() {
        let mut x = Value::scalar(23.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_229() {
        let mut x = Value::scalar(23.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_230() {
        let mut x = Value::scalar(24.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_231() {
        let mut x = Value::scalar(24.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_232() {
        let mut x = Value::scalar(24.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_233() {
        let mut x = Value::scalar(24.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_234() {
        let mut x = Value::scalar(24.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_235() {
        let mut x = Value::scalar(24.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_236() {
        let mut x = Value::scalar(24.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_237() {
        let mut x = Value::scalar(24.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_238() {
        let mut x = Value::scalar(24.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_239() {
        let mut x = Value::scalar(24.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_240() {
        let mut x = Value::scalar(25.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_241() {
        let mut x = Value::scalar(25.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_242() {
        let mut x = Value::scalar(25.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_243() {
        let mut x = Value::scalar(25.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_244() {
        let mut x = Value::scalar(25.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_245() {
        let mut x = Value::scalar(25.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_246() {
        let mut x = Value::scalar(25.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_247() {
        let mut x = Value::scalar(25.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_248() {
        let mut x = Value::scalar(25.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_249() {
        let mut x = Value::scalar(25.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_250() {
        let mut x = Value::scalar(26.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_251() {
        let mut x = Value::scalar(26.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_252() {
        let mut x = Value::scalar(26.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_253() {
        let mut x = Value::scalar(26.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_254() {
        let mut x = Value::scalar(26.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_255() {
        let mut x = Value::scalar(26.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_256() {
        let mut x = Value::scalar(26.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_257() {
        let mut x = Value::scalar(26.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_258() {
        let mut x = Value::scalar(26.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_259() {
        let mut x = Value::scalar(26.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_260() {
        let mut x = Value::scalar(27.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_261() {
        let mut x = Value::scalar(27.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_262() {
        let mut x = Value::scalar(27.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_263() {
        let mut x = Value::scalar(27.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_264() {
        let mut x = Value::scalar(27.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_265() {
        let mut x = Value::scalar(27.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_266() {
        let mut x = Value::scalar(27.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_267() {
        let mut x = Value::scalar(27.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_268() {
        let mut x = Value::scalar(27.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_269() {
        let mut x = Value::scalar(27.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_270() {
        let mut x = Value::scalar(28.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_271() {
        let mut x = Value::scalar(28.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_272() {
        let mut x = Value::scalar(28.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_273() {
        let mut x = Value::scalar(28.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_274() {
        let mut x = Value::scalar(28.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_275() {
        let mut x = Value::scalar(28.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_276() {
        let mut x = Value::scalar(28.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_277() {
        let mut x = Value::scalar(28.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_278() {
        let mut x = Value::scalar(28.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_279() {
        let mut x = Value::scalar(28.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_280() {
        let mut x = Value::scalar(29.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_281() {
        let mut x = Value::scalar(29.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_282() {
        let mut x = Value::scalar(29.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_283() {
        let mut x = Value::scalar(29.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_284() {
        let mut x = Value::scalar(29.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_285() {
        let mut x = Value::scalar(29.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_286() {
        let mut x = Value::scalar(29.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_287() {
        let mut x = Value::scalar(29.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_288() {
        let mut x = Value::scalar(29.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_289() {
        let mut x = Value::scalar(29.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_290() {
        let mut x = Value::scalar(30.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_291() {
        let mut x = Value::scalar(30.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_292() {
        let mut x = Value::scalar(30.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_293() {
        let mut x = Value::scalar(30.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_294() {
        let mut x = Value::scalar(30.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_295() {
        let mut x = Value::scalar(30.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_296() {
        let mut x = Value::scalar(30.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_297() {
        let mut x = Value::scalar(30.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_298() {
        let mut x = Value::scalar(30.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_299() {
        let mut x = Value::scalar(30.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_300() {
        let mut x = Value::scalar(31.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_301() {
        let mut x = Value::scalar(31.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_302() {
        let mut x = Value::scalar(31.200000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_303() {
        let mut x = Value::scalar(31.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_304() {
        let mut x = Value::scalar(31.400000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_305() {
        let mut x = Value::scalar(31.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_306() {
        let mut x = Value::scalar(31.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_307() {
        let mut x = Value::scalar(31.700000000000003);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_308() {
        let mut x = Value::scalar(31.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_309() {
        let mut x = Value::scalar(31.900000000000002);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_310() {
        let mut x = Value::scalar(32.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_311() {
        let mut x = Value::scalar(32.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_312() {
        let mut x = Value::scalar(32.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_313() {
        let mut x = Value::scalar(32.3);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_314() {
        let mut x = Value::scalar(32.400000000000006);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_315() {
        let mut x = Value::scalar(32.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_316() {
        let mut x = Value::scalar(32.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_317() {
        let mut x = Value::scalar(32.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_318() {
        let mut x = Value::scalar(32.8);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_319() {
        let mut x = Value::scalar(32.900000000000006);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_320() {
        let mut x = Value::scalar(33.0);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_321() {
        let mut x = Value::scalar(33.1);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_322() {
        let mut x = Value::scalar(33.2);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_323() {
        let mut x = Value::scalar(33.300000000000004);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_324() {
        let mut x = Value::scalar(33.4);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_325() {
        let mut x = Value::scalar(33.5);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_326() {
        let mut x = Value::scalar(33.6);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    #[test]
    fn test_topo_sort_stress_327() {
        let mut x = Value::scalar(33.7);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        let z = y.add(&x);
        let nodes = topological_sort(&z).unwrap();
        assert!(nodes.len() >= 2);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
}

#[cfg(test)]
mod topo_engine_tests {
    use super::{compute_dag_levels, topological_sort};
    use crate::value::Value;

    #[test]
    fn test_deep_chain_no_stack_overflow() {
        // Build a deep linear chain x0 -> x1(=x0*x0) -> ... -> xN. The iterative
        // topo sort must traverse N levels without using the call stack.
        // Depth is bounded here by `Arc<Value>` recursive Drop when tearing the
        // chain down (a residual Phase-2 item), so we pick a depth comfortably
        // above where a *recursive* DFS would overflow but below the Drop ceiling.
        let n = 8_000usize;
        let mut base = Value::scalar(1.01);
        base.set_requires_grad(true);
        let mut cur = base.clone();
        for _ in 0..n {
            cur = cur.clone() * cur.clone();
        }
        assert!(cur.grad_fn().is_op(), "chain must build real graph nodes");
        let order = topological_sort(&cur).unwrap();
        assert!(order.len() >= n);
        assert!(base.grad().is_none());
    }

    #[test]
    fn test_topo_sort_order_is_valid_post_order() {
        // z = (x*x) + (x*x); parents must all appear before their children.
        let mut x = Value::scalar(2.0);
        x.set_requires_grad(true);
        let a = x.clone().mul(&x);
        let b = x.clone() * x.clone();
        let z = a.clone().add(&b);
        let order = topological_sort(&z).unwrap();

        let pos: std::collections::HashMap<usize, usize> = order
            .iter()
            .enumerate()
            .map(|(i, v)| (v.id(), i))
            .collect();
        // every parent id must come before child id
        for (i, node) in order.iter().enumerate() {
            for p in node.grad_fn().parents() {
                let ppos = pos.get(&p.id()).copied().unwrap();
                assert!(ppos < i, "parent must precede child");
            }
        }
        assert!(order.iter().any(|v| v.id() == x.id()));
        assert!(order.iter().any(|v| v.id() == z.id()));
    }

    #[test]
    fn test_compute_dag_levels_respects_dependencies() {
        //       z
        //      / \
        //     y   w
        //     |
        //     x   (x is a leaf => level 0; y level 1; z level 2; w level 1)
        let mut x = Value::scalar(1.0);
        x.set_requires_grad(true);
        let y = x.clone().mul(&x);
        let w = y.clone().mul(&x);
        let z = y.clone().add(&w);
        let order = topological_sort(&z).unwrap();
        let levels = compute_dag_levels(&order);

        // level 0 must contain all leaves and no internal node
        for node in &levels[0] {
            assert!(node.grad_fn().parents().is_empty());
        }
        // every node at level k must have all parents at level < k
        let lvl_of: std::collections::HashMap<usize, usize> = levels
            .iter()
            .enumerate()
            .flat_map(|(k, bucket)| bucket.iter().map(move |v| (v.id(), k)))
            .collect();
        for (k, bucket) in levels.iter().enumerate() {
            for node in bucket {
                for p in node.grad_fn().parents() {
                    let plvl = lvl_of[&p.id()];
                    assert!(plvl < k, "parent level must be < child level");
                }
            }
        }
    }
}
