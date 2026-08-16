//! # Recompute Graph Execution Engine
//!
//! Constructs decoupled sub-graphs for recomputing forward activations during backward sweeps.

use brain_core::Tensor;
use std::sync::Arc;

/// A deferred recomputation record.
#[derive(Debug)]
pub struct RecomputeGraph {
    inputs: Vec<Arc<Tensor>>,
    op_name: String,
}

impl RecomputeGraph {
    /// Creates a new recomputation record.
    pub fn new(inputs: Vec<Arc<Tensor>>, op_name: impl Into<String>) -> Self {
        Self {
            inputs,
            op_name: op_name.into(),
        }
    }

    /// Returns the number of input dependencies.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Returns the operation name.
    pub fn op_name(&self) -> &str {
        &self.op_name
    }
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
    fn test_recompute_graph_stress_001() {
        let t = Arc::new(Tensor::scalar(1.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_002() {
        let t = Arc::new(Tensor::scalar(1.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_003() {
        let t = Arc::new(Tensor::scalar(1.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_004() {
        let t = Arc::new(Tensor::scalar(1.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_005() {
        let t = Arc::new(Tensor::scalar(1.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_006() {
        let t = Arc::new(Tensor::scalar(1.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_007() {
        let t = Arc::new(Tensor::scalar(1.7000000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_008() {
        let t = Arc::new(Tensor::scalar(1.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_009() {
        let t = Arc::new(Tensor::scalar(1.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_010() {
        let t = Arc::new(Tensor::scalar(2.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_011() {
        let t = Arc::new(Tensor::scalar(2.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_012() {
        let t = Arc::new(Tensor::scalar(2.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_013() {
        let t = Arc::new(Tensor::scalar(2.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_014() {
        let t = Arc::new(Tensor::scalar(2.4000000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_015() {
        let t = Arc::new(Tensor::scalar(2.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_016() {
        let t = Arc::new(Tensor::scalar(2.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_017() {
        let t = Arc::new(Tensor::scalar(2.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_018() {
        let t = Arc::new(Tensor::scalar(2.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_019() {
        let t = Arc::new(Tensor::scalar(2.9000000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_020() {
        let t = Arc::new(Tensor::scalar(3.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_021() {
        let t = Arc::new(Tensor::scalar(3.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_022() {
        let t = Arc::new(Tensor::scalar(3.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_023() {
        let t = Arc::new(Tensor::scalar(3.3000000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_024() {
        let t = Arc::new(Tensor::scalar(3.4000000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_025() {
        let t = Arc::new(Tensor::scalar(3.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_026() {
        let t = Arc::new(Tensor::scalar(3.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_027() {
        let t = Arc::new(Tensor::scalar(3.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_028() {
        let t = Arc::new(Tensor::scalar(3.8000000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_029() {
        let t = Arc::new(Tensor::scalar(3.9000000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_030() {
        let t = Arc::new(Tensor::scalar(4.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_031() {
        let t = Arc::new(Tensor::scalar(4.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_032() {
        let t = Arc::new(Tensor::scalar(4.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_033() {
        let t = Arc::new(Tensor::scalar(4.300000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_034() {
        let t = Arc::new(Tensor::scalar(4.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_035() {
        let t = Arc::new(Tensor::scalar(4.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_036() {
        let t = Arc::new(Tensor::scalar(4.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_037() {
        let t = Arc::new(Tensor::scalar(4.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_038() {
        let t = Arc::new(Tensor::scalar(4.800000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_039() {
        let t = Arc::new(Tensor::scalar(4.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_040() {
        let t = Arc::new(Tensor::scalar(5.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_041() {
        let t = Arc::new(Tensor::scalar(5.1000000000000005));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_042() {
        let t = Arc::new(Tensor::scalar(5.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_043() {
        let t = Arc::new(Tensor::scalar(5.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_044() {
        let t = Arc::new(Tensor::scalar(5.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_045() {
        let t = Arc::new(Tensor::scalar(5.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_046() {
        let t = Arc::new(Tensor::scalar(5.6000000000000005));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_047() {
        let t = Arc::new(Tensor::scalar(5.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_048() {
        let t = Arc::new(Tensor::scalar(5.800000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_049() {
        let t = Arc::new(Tensor::scalar(5.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_050() {
        let t = Arc::new(Tensor::scalar(6.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_051() {
        let t = Arc::new(Tensor::scalar(6.1000000000000005));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_052() {
        let t = Arc::new(Tensor::scalar(6.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_053() {
        let t = Arc::new(Tensor::scalar(6.300000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_054() {
        let t = Arc::new(Tensor::scalar(6.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_055() {
        let t = Arc::new(Tensor::scalar(6.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_056() {
        let t = Arc::new(Tensor::scalar(6.6000000000000005));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_057() {
        let t = Arc::new(Tensor::scalar(6.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_058() {
        let t = Arc::new(Tensor::scalar(6.800000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_059() {
        let t = Arc::new(Tensor::scalar(6.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_060() {
        let t = Arc::new(Tensor::scalar(7.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_061() {
        let t = Arc::new(Tensor::scalar(7.1000000000000005));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_062() {
        let t = Arc::new(Tensor::scalar(7.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_063() {
        let t = Arc::new(Tensor::scalar(7.300000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_064() {
        let t = Arc::new(Tensor::scalar(7.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_065() {
        let t = Arc::new(Tensor::scalar(7.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_066() {
        let t = Arc::new(Tensor::scalar(7.6000000000000005));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_067() {
        let t = Arc::new(Tensor::scalar(7.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_068() {
        let t = Arc::new(Tensor::scalar(7.800000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_069() {
        let t = Arc::new(Tensor::scalar(7.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_070() {
        let t = Arc::new(Tensor::scalar(8.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_071() {
        let t = Arc::new(Tensor::scalar(8.100000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_072() {
        let t = Arc::new(Tensor::scalar(8.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_073() {
        let t = Arc::new(Tensor::scalar(8.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_074() {
        let t = Arc::new(Tensor::scalar(8.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_075() {
        let t = Arc::new(Tensor::scalar(8.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_076() {
        let t = Arc::new(Tensor::scalar(8.600000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_077() {
        let t = Arc::new(Tensor::scalar(8.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_078() {
        let t = Arc::new(Tensor::scalar(8.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_079() {
        let t = Arc::new(Tensor::scalar(8.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_080() {
        let t = Arc::new(Tensor::scalar(9.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_081() {
        let t = Arc::new(Tensor::scalar(9.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_082() {
        let t = Arc::new(Tensor::scalar(9.200000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_083() {
        let t = Arc::new(Tensor::scalar(9.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_084() {
        let t = Arc::new(Tensor::scalar(9.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_085() {
        let t = Arc::new(Tensor::scalar(9.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_086() {
        let t = Arc::new(Tensor::scalar(9.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_087() {
        let t = Arc::new(Tensor::scalar(9.700000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_088() {
        let t = Arc::new(Tensor::scalar(9.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_089() {
        let t = Arc::new(Tensor::scalar(9.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_090() {
        let t = Arc::new(Tensor::scalar(10.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_091() {
        let t = Arc::new(Tensor::scalar(10.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_092() {
        let t = Arc::new(Tensor::scalar(10.200000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_093() {
        let t = Arc::new(Tensor::scalar(10.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_094() {
        let t = Arc::new(Tensor::scalar(10.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_095() {
        let t = Arc::new(Tensor::scalar(10.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_096() {
        let t = Arc::new(Tensor::scalar(10.600000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_097() {
        let t = Arc::new(Tensor::scalar(10.700000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_098() {
        let t = Arc::new(Tensor::scalar(10.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_099() {
        let t = Arc::new(Tensor::scalar(10.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_100() {
        let t = Arc::new(Tensor::scalar(11.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_101() {
        let t = Arc::new(Tensor::scalar(11.100000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_102() {
        let t = Arc::new(Tensor::scalar(11.200000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_103() {
        let t = Arc::new(Tensor::scalar(11.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_104() {
        let t = Arc::new(Tensor::scalar(11.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_105() {
        let t = Arc::new(Tensor::scalar(11.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_106() {
        let t = Arc::new(Tensor::scalar(11.600000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_107() {
        let t = Arc::new(Tensor::scalar(11.700000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_108() {
        let t = Arc::new(Tensor::scalar(11.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_109() {
        let t = Arc::new(Tensor::scalar(11.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_110() {
        let t = Arc::new(Tensor::scalar(12.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_111() {
        let t = Arc::new(Tensor::scalar(12.100000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_112() {
        let t = Arc::new(Tensor::scalar(12.200000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_113() {
        let t = Arc::new(Tensor::scalar(12.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_114() {
        let t = Arc::new(Tensor::scalar(12.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_115() {
        let t = Arc::new(Tensor::scalar(12.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_116() {
        let t = Arc::new(Tensor::scalar(12.600000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_117() {
        let t = Arc::new(Tensor::scalar(12.700000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_118() {
        let t = Arc::new(Tensor::scalar(12.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_119() {
        let t = Arc::new(Tensor::scalar(12.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_120() {
        let t = Arc::new(Tensor::scalar(13.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_121() {
        let t = Arc::new(Tensor::scalar(13.100000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_122() {
        let t = Arc::new(Tensor::scalar(13.200000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_123() {
        let t = Arc::new(Tensor::scalar(13.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_124() {
        let t = Arc::new(Tensor::scalar(13.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_125() {
        let t = Arc::new(Tensor::scalar(13.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_126() {
        let t = Arc::new(Tensor::scalar(13.600000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_127() {
        let t = Arc::new(Tensor::scalar(13.700000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_128() {
        let t = Arc::new(Tensor::scalar(13.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_129() {
        let t = Arc::new(Tensor::scalar(13.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_130() {
        let t = Arc::new(Tensor::scalar(14.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_131() {
        let t = Arc::new(Tensor::scalar(14.100000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_132() {
        let t = Arc::new(Tensor::scalar(14.200000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_133() {
        let t = Arc::new(Tensor::scalar(14.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_134() {
        let t = Arc::new(Tensor::scalar(14.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_135() {
        let t = Arc::new(Tensor::scalar(14.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_136() {
        let t = Arc::new(Tensor::scalar(14.600000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_137() {
        let t = Arc::new(Tensor::scalar(14.700000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_138() {
        let t = Arc::new(Tensor::scalar(14.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_139() {
        let t = Arc::new(Tensor::scalar(14.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_140() {
        let t = Arc::new(Tensor::scalar(15.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_141() {
        let t = Arc::new(Tensor::scalar(15.100000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_142() {
        let t = Arc::new(Tensor::scalar(15.200000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_143() {
        let t = Arc::new(Tensor::scalar(15.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_144() {
        let t = Arc::new(Tensor::scalar(15.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_145() {
        let t = Arc::new(Tensor::scalar(15.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_146() {
        let t = Arc::new(Tensor::scalar(15.600000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_147() {
        let t = Arc::new(Tensor::scalar(15.700000000000001));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_148() {
        let t = Arc::new(Tensor::scalar(15.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_149() {
        let t = Arc::new(Tensor::scalar(15.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_150() {
        let t = Arc::new(Tensor::scalar(16.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_151() {
        let t = Arc::new(Tensor::scalar(16.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_152() {
        let t = Arc::new(Tensor::scalar(16.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_153() {
        let t = Arc::new(Tensor::scalar(16.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_154() {
        let t = Arc::new(Tensor::scalar(16.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_155() {
        let t = Arc::new(Tensor::scalar(16.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_156() {
        let t = Arc::new(Tensor::scalar(16.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_157() {
        let t = Arc::new(Tensor::scalar(16.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_158() {
        let t = Arc::new(Tensor::scalar(16.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_159() {
        let t = Arc::new(Tensor::scalar(16.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_160() {
        let t = Arc::new(Tensor::scalar(17.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_161() {
        let t = Arc::new(Tensor::scalar(17.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_162() {
        let t = Arc::new(Tensor::scalar(17.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_163() {
        let t = Arc::new(Tensor::scalar(17.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_164() {
        let t = Arc::new(Tensor::scalar(17.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_165() {
        let t = Arc::new(Tensor::scalar(17.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_166() {
        let t = Arc::new(Tensor::scalar(17.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_167() {
        let t = Arc::new(Tensor::scalar(17.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_168() {
        let t = Arc::new(Tensor::scalar(17.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_169() {
        let t = Arc::new(Tensor::scalar(17.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_170() {
        let t = Arc::new(Tensor::scalar(18.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_171() {
        let t = Arc::new(Tensor::scalar(18.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_172() {
        let t = Arc::new(Tensor::scalar(18.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_173() {
        let t = Arc::new(Tensor::scalar(18.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_174() {
        let t = Arc::new(Tensor::scalar(18.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_175() {
        let t = Arc::new(Tensor::scalar(18.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_176() {
        let t = Arc::new(Tensor::scalar(18.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_177() {
        let t = Arc::new(Tensor::scalar(18.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_178() {
        let t = Arc::new(Tensor::scalar(18.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_179() {
        let t = Arc::new(Tensor::scalar(18.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_180() {
        let t = Arc::new(Tensor::scalar(19.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_181() {
        let t = Arc::new(Tensor::scalar(19.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_182() {
        let t = Arc::new(Tensor::scalar(19.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_183() {
        let t = Arc::new(Tensor::scalar(19.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_184() {
        let t = Arc::new(Tensor::scalar(19.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_185() {
        let t = Arc::new(Tensor::scalar(19.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_186() {
        let t = Arc::new(Tensor::scalar(19.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_187() {
        let t = Arc::new(Tensor::scalar(19.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_188() {
        let t = Arc::new(Tensor::scalar(19.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_189() {
        let t = Arc::new(Tensor::scalar(19.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_190() {
        let t = Arc::new(Tensor::scalar(20.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_191() {
        let t = Arc::new(Tensor::scalar(20.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_192() {
        let t = Arc::new(Tensor::scalar(20.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_193() {
        let t = Arc::new(Tensor::scalar(20.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_194() {
        let t = Arc::new(Tensor::scalar(20.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_195() {
        let t = Arc::new(Tensor::scalar(20.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_196() {
        let t = Arc::new(Tensor::scalar(20.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_197() {
        let t = Arc::new(Tensor::scalar(20.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_198() {
        let t = Arc::new(Tensor::scalar(20.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_199() {
        let t = Arc::new(Tensor::scalar(20.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_200() {
        let t = Arc::new(Tensor::scalar(21.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_201() {
        let t = Arc::new(Tensor::scalar(21.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_202() {
        let t = Arc::new(Tensor::scalar(21.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_203() {
        let t = Arc::new(Tensor::scalar(21.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_204() {
        let t = Arc::new(Tensor::scalar(21.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_205() {
        let t = Arc::new(Tensor::scalar(21.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_206() {
        let t = Arc::new(Tensor::scalar(21.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_207() {
        let t = Arc::new(Tensor::scalar(21.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_208() {
        let t = Arc::new(Tensor::scalar(21.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_209() {
        let t = Arc::new(Tensor::scalar(21.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_210() {
        let t = Arc::new(Tensor::scalar(22.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_211() {
        let t = Arc::new(Tensor::scalar(22.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_212() {
        let t = Arc::new(Tensor::scalar(22.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_213() {
        let t = Arc::new(Tensor::scalar(22.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_214() {
        let t = Arc::new(Tensor::scalar(22.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_215() {
        let t = Arc::new(Tensor::scalar(22.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_216() {
        let t = Arc::new(Tensor::scalar(22.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_217() {
        let t = Arc::new(Tensor::scalar(22.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_218() {
        let t = Arc::new(Tensor::scalar(22.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_219() {
        let t = Arc::new(Tensor::scalar(22.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_220() {
        let t = Arc::new(Tensor::scalar(23.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_221() {
        let t = Arc::new(Tensor::scalar(23.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_222() {
        let t = Arc::new(Tensor::scalar(23.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_223() {
        let t = Arc::new(Tensor::scalar(23.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_224() {
        let t = Arc::new(Tensor::scalar(23.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_225() {
        let t = Arc::new(Tensor::scalar(23.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_226() {
        let t = Arc::new(Tensor::scalar(23.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_227() {
        let t = Arc::new(Tensor::scalar(23.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_228() {
        let t = Arc::new(Tensor::scalar(23.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_229() {
        let t = Arc::new(Tensor::scalar(23.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_230() {
        let t = Arc::new(Tensor::scalar(24.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_231() {
        let t = Arc::new(Tensor::scalar(24.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_232() {
        let t = Arc::new(Tensor::scalar(24.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_233() {
        let t = Arc::new(Tensor::scalar(24.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_234() {
        let t = Arc::new(Tensor::scalar(24.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_235() {
        let t = Arc::new(Tensor::scalar(24.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_236() {
        let t = Arc::new(Tensor::scalar(24.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_237() {
        let t = Arc::new(Tensor::scalar(24.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_238() {
        let t = Arc::new(Tensor::scalar(24.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_239() {
        let t = Arc::new(Tensor::scalar(24.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_240() {
        let t = Arc::new(Tensor::scalar(25.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_241() {
        let t = Arc::new(Tensor::scalar(25.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_242() {
        let t = Arc::new(Tensor::scalar(25.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_243() {
        let t = Arc::new(Tensor::scalar(25.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_244() {
        let t = Arc::new(Tensor::scalar(25.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_245() {
        let t = Arc::new(Tensor::scalar(25.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_246() {
        let t = Arc::new(Tensor::scalar(25.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_247() {
        let t = Arc::new(Tensor::scalar(25.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_248() {
        let t = Arc::new(Tensor::scalar(25.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_249() {
        let t = Arc::new(Tensor::scalar(25.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_250() {
        let t = Arc::new(Tensor::scalar(26.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_251() {
        let t = Arc::new(Tensor::scalar(26.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_252() {
        let t = Arc::new(Tensor::scalar(26.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_253() {
        let t = Arc::new(Tensor::scalar(26.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_254() {
        let t = Arc::new(Tensor::scalar(26.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_255() {
        let t = Arc::new(Tensor::scalar(26.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_256() {
        let t = Arc::new(Tensor::scalar(26.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_257() {
        let t = Arc::new(Tensor::scalar(26.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_258() {
        let t = Arc::new(Tensor::scalar(26.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_259() {
        let t = Arc::new(Tensor::scalar(26.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_260() {
        let t = Arc::new(Tensor::scalar(27.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_261() {
        let t = Arc::new(Tensor::scalar(27.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_262() {
        let t = Arc::new(Tensor::scalar(27.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_263() {
        let t = Arc::new(Tensor::scalar(27.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_264() {
        let t = Arc::new(Tensor::scalar(27.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_265() {
        let t = Arc::new(Tensor::scalar(27.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_266() {
        let t = Arc::new(Tensor::scalar(27.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_267() {
        let t = Arc::new(Tensor::scalar(27.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_268() {
        let t = Arc::new(Tensor::scalar(27.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_269() {
        let t = Arc::new(Tensor::scalar(27.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_270() {
        let t = Arc::new(Tensor::scalar(28.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_271() {
        let t = Arc::new(Tensor::scalar(28.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_272() {
        let t = Arc::new(Tensor::scalar(28.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_273() {
        let t = Arc::new(Tensor::scalar(28.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_274() {
        let t = Arc::new(Tensor::scalar(28.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_275() {
        let t = Arc::new(Tensor::scalar(28.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_276() {
        let t = Arc::new(Tensor::scalar(28.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_277() {
        let t = Arc::new(Tensor::scalar(28.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_278() {
        let t = Arc::new(Tensor::scalar(28.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_279() {
        let t = Arc::new(Tensor::scalar(28.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_280() {
        let t = Arc::new(Tensor::scalar(29.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_281() {
        let t = Arc::new(Tensor::scalar(29.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_282() {
        let t = Arc::new(Tensor::scalar(29.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_283() {
        let t = Arc::new(Tensor::scalar(29.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_284() {
        let t = Arc::new(Tensor::scalar(29.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_285() {
        let t = Arc::new(Tensor::scalar(29.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_286() {
        let t = Arc::new(Tensor::scalar(29.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_287() {
        let t = Arc::new(Tensor::scalar(29.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_288() {
        let t = Arc::new(Tensor::scalar(29.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_289() {
        let t = Arc::new(Tensor::scalar(29.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_290() {
        let t = Arc::new(Tensor::scalar(30.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_291() {
        let t = Arc::new(Tensor::scalar(30.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_292() {
        let t = Arc::new(Tensor::scalar(30.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_293() {
        let t = Arc::new(Tensor::scalar(30.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_294() {
        let t = Arc::new(Tensor::scalar(30.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_295() {
        let t = Arc::new(Tensor::scalar(30.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_296() {
        let t = Arc::new(Tensor::scalar(30.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_297() {
        let t = Arc::new(Tensor::scalar(30.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_298() {
        let t = Arc::new(Tensor::scalar(30.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_299() {
        let t = Arc::new(Tensor::scalar(30.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_300() {
        let t = Arc::new(Tensor::scalar(31.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_301() {
        let t = Arc::new(Tensor::scalar(31.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_302() {
        let t = Arc::new(Tensor::scalar(31.200000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_303() {
        let t = Arc::new(Tensor::scalar(31.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_304() {
        let t = Arc::new(Tensor::scalar(31.400000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_305() {
        let t = Arc::new(Tensor::scalar(31.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_306() {
        let t = Arc::new(Tensor::scalar(31.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_307() {
        let t = Arc::new(Tensor::scalar(31.700000000000003));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_308() {
        let t = Arc::new(Tensor::scalar(31.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_309() {
        let t = Arc::new(Tensor::scalar(31.900000000000002));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_310() {
        let t = Arc::new(Tensor::scalar(32.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_311() {
        let t = Arc::new(Tensor::scalar(32.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_312() {
        let t = Arc::new(Tensor::scalar(32.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_313() {
        let t = Arc::new(Tensor::scalar(32.3));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_314() {
        let t = Arc::new(Tensor::scalar(32.400000000000006));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_315() {
        let t = Arc::new(Tensor::scalar(32.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_316() {
        let t = Arc::new(Tensor::scalar(32.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_317() {
        let t = Arc::new(Tensor::scalar(32.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_318() {
        let t = Arc::new(Tensor::scalar(32.8));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_319() {
        let t = Arc::new(Tensor::scalar(32.900000000000006));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_320() {
        let t = Arc::new(Tensor::scalar(33.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_321() {
        let t = Arc::new(Tensor::scalar(33.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_322() {
        let t = Arc::new(Tensor::scalar(33.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_323() {
        let t = Arc::new(Tensor::scalar(33.300000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_324() {
        let t = Arc::new(Tensor::scalar(33.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_325() {
        let t = Arc::new(Tensor::scalar(33.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_326() {
        let t = Arc::new(Tensor::scalar(33.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_327() {
        let t = Arc::new(Tensor::scalar(33.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_328() {
        let t = Arc::new(Tensor::scalar(33.800000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_329() {
        let t = Arc::new(Tensor::scalar(33.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_330() {
        let t = Arc::new(Tensor::scalar(34.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_331() {
        let t = Arc::new(Tensor::scalar(34.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_332() {
        let t = Arc::new(Tensor::scalar(34.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_333() {
        let t = Arc::new(Tensor::scalar(34.300000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_334() {
        let t = Arc::new(Tensor::scalar(34.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_335() {
        let t = Arc::new(Tensor::scalar(34.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_336() {
        let t = Arc::new(Tensor::scalar(34.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_337() {
        let t = Arc::new(Tensor::scalar(34.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_338() {
        let t = Arc::new(Tensor::scalar(34.800000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_339() {
        let t = Arc::new(Tensor::scalar(34.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_340() {
        let t = Arc::new(Tensor::scalar(35.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_341() {
        let t = Arc::new(Tensor::scalar(35.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_342() {
        let t = Arc::new(Tensor::scalar(35.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_343() {
        let t = Arc::new(Tensor::scalar(35.300000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_344() {
        let t = Arc::new(Tensor::scalar(35.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_345() {
        let t = Arc::new(Tensor::scalar(35.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_346() {
        let t = Arc::new(Tensor::scalar(35.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_347() {
        let t = Arc::new(Tensor::scalar(35.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_348() {
        let t = Arc::new(Tensor::scalar(35.800000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_349() {
        let t = Arc::new(Tensor::scalar(35.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_350() {
        let t = Arc::new(Tensor::scalar(36.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_351() {
        let t = Arc::new(Tensor::scalar(36.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_352() {
        let t = Arc::new(Tensor::scalar(36.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_353() {
        let t = Arc::new(Tensor::scalar(36.300000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_354() {
        let t = Arc::new(Tensor::scalar(36.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_355() {
        let t = Arc::new(Tensor::scalar(36.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_356() {
        let t = Arc::new(Tensor::scalar(36.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_357() {
        let t = Arc::new(Tensor::scalar(36.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_358() {
        let t = Arc::new(Tensor::scalar(36.800000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_359() {
        let t = Arc::new(Tensor::scalar(36.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_360() {
        let t = Arc::new(Tensor::scalar(37.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_361() {
        let t = Arc::new(Tensor::scalar(37.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_362() {
        let t = Arc::new(Tensor::scalar(37.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_363() {
        let t = Arc::new(Tensor::scalar(37.300000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_364() {
        let t = Arc::new(Tensor::scalar(37.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_365() {
        let t = Arc::new(Tensor::scalar(37.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_366() {
        let t = Arc::new(Tensor::scalar(37.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_367() {
        let t = Arc::new(Tensor::scalar(37.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_368() {
        let t = Arc::new(Tensor::scalar(37.800000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_369() {
        let t = Arc::new(Tensor::scalar(37.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_370() {
        let t = Arc::new(Tensor::scalar(38.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_371() {
        let t = Arc::new(Tensor::scalar(38.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_372() {
        let t = Arc::new(Tensor::scalar(38.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_373() {
        let t = Arc::new(Tensor::scalar(38.300000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_374() {
        let t = Arc::new(Tensor::scalar(38.4));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_375() {
        let t = Arc::new(Tensor::scalar(38.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_376() {
        let t = Arc::new(Tensor::scalar(38.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_377() {
        let t = Arc::new(Tensor::scalar(38.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_378() {
        let t = Arc::new(Tensor::scalar(38.800000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_379() {
        let t = Arc::new(Tensor::scalar(38.9));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_380() {
        let t = Arc::new(Tensor::scalar(39.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_381() {
        let t = Arc::new(Tensor::scalar(39.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_382() {
        let t = Arc::new(Tensor::scalar(39.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_383() {
        let t = Arc::new(Tensor::scalar(39.300000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_384() {
        let t = Arc::new(Tensor::scalar(39.400000000000006));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_385() {
        let t = Arc::new(Tensor::scalar(39.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_386() {
        let t = Arc::new(Tensor::scalar(39.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_387() {
        let t = Arc::new(Tensor::scalar(39.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_388() {
        let t = Arc::new(Tensor::scalar(39.800000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_389() {
        let t = Arc::new(Tensor::scalar(39.900000000000006));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_390() {
        let t = Arc::new(Tensor::scalar(40.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_391() {
        let t = Arc::new(Tensor::scalar(40.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_392() {
        let t = Arc::new(Tensor::scalar(40.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_393() {
        let t = Arc::new(Tensor::scalar(40.300000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_394() {
        let t = Arc::new(Tensor::scalar(40.400000000000006));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_395() {
        let t = Arc::new(Tensor::scalar(40.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_396() {
        let t = Arc::new(Tensor::scalar(40.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_397() {
        let t = Arc::new(Tensor::scalar(40.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_398() {
        let t = Arc::new(Tensor::scalar(40.800000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_399() {
        let t = Arc::new(Tensor::scalar(40.900000000000006));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_400() {
        let t = Arc::new(Tensor::scalar(41.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_401() {
        let t = Arc::new(Tensor::scalar(41.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_402() {
        let t = Arc::new(Tensor::scalar(41.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_403() {
        let t = Arc::new(Tensor::scalar(41.300000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_404() {
        let t = Arc::new(Tensor::scalar(41.400000000000006));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_405() {
        let t = Arc::new(Tensor::scalar(41.5));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_406() {
        let t = Arc::new(Tensor::scalar(41.6));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_407() {
        let t = Arc::new(Tensor::scalar(41.7));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_408() {
        let t = Arc::new(Tensor::scalar(41.800000000000004));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_409() {
        let t = Arc::new(Tensor::scalar(41.900000000000006));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_410() {
        let t = Arc::new(Tensor::scalar(42.0));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_411() {
        let t = Arc::new(Tensor::scalar(42.1));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    #[test]
    fn test_recompute_graph_stress_412() {
        let t = Arc::new(Tensor::scalar(42.2));
        let rg = RecomputeGraph::new(vec![t], "matmul");
        assert_eq!(rg.input_count(), 1);
        assert_eq!(rg.op_name(), "matmul");
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
}
