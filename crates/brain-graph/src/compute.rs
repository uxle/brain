//! # Cost Model & Metric Computations
//!
//! Theoretical FLOP/MAC, parameter size, and arithmetic intensity calculation.
#![allow(missing_docs)]

use crate::ir::GraphIr;

/// Detailed computational cost statistics.
#[derive(Debug, Clone, Default)]
pub struct GraphCosts {
    pub total_flops: usize,
    pub total_parameters: usize,
    pub total_memory_traffic_bytes: usize,
}

/// Computes comprehensive costs for a `GraphIr`.
pub fn compute_costs(graph: &GraphIr) -> GraphCosts {
    let mut total_flops = 0;
    let mut total_params = 0;
    let mut total_bytes = 0;

    for v in &graph.values {
        let n = v.shape.num_elements();
        if v.constant_data.is_some() {
            total_params += n;
        }
        total_bytes += n * 4;
    }

    for node in &graph.nodes {
        if let Some(&out) = node.outputs.first() {
            let n = graph.values[out].shape.num_elements();
            total_flops += match node.op {
                crate::ir::ops::OpKind::MatMul => n * 2,
                _ => n,
            };
        }
    }

    GraphCosts {
        total_flops,
        total_parameters: total_params,
        total_memory_traffic_bytes: total_bytes,
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
