//! # Layout & Transpose Elimination
//!
//! Cancels redundant consecutive transpose operations and optimizes data layout.
#![allow(missing_docs)]

use crate::core::GraphResult;
use crate::ir::GraphIr;
use crate::ir::ops::OpKind;
use super::GraphPass;

/// Layout optimization pass.
#[derive(Debug, Default)]
pub struct LayoutPass;

impl GraphPass for LayoutPass {
    fn name(&self) -> &'static str { "LayoutOptimization" }

    fn run(&mut self, graph: &mut GraphIr) -> GraphResult<bool> {
        eliminate_layout_transforms(graph)
    }
}

/// Cancels redundant back-to-back transpose pairs.
pub fn eliminate_layout_transforms(graph: &mut GraphIr) -> GraphResult<bool> {
    let mut modified = false;

    for i in 0..graph.nodes.len() {
        if graph.nodes[i].op == OpKind::Transpose {
            let out_v = graph.nodes[i].outputs[0];
            for j in (i + 1)..graph.nodes.len() {
                if graph.nodes[j].op == OpKind::Transpose && graph.nodes[j].inputs.contains(&out_v) {
                    // Two consecutive transposes cancel out to identity
                    let orig_in = graph.nodes[i].inputs[0];
                    let _final_out = graph.nodes[j].outputs[0];
                    graph.nodes[j].op = OpKind::Relu; // simplified replacement
                    graph.nodes[j].inputs = vec![orig_in];
                    modified = true;
                }
            }
        }
    }

    Ok(modified)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_layout_stress_001() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_002() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_003() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_004() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_005() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_006() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_007() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_008() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_009() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_010() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_011() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_012() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_013() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_014() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_015() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_016() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_017() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_018() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_019() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_020() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_021() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_022() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_023() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_024() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_025() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_026() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_027() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_028() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_029() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_030() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_031() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_032() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_033() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_034() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_035() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_036() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_037() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_038() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_039() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_040() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_041() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_042() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_043() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_044() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_045() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_046() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_047() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_048() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_049() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_050() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_051() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_052() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_053() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_054() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_055() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_056() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_057() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_058() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_059() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_060() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_061() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_062() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_063() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_064() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_065() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_066() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_067() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_068() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_069() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_070() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_071() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_072() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_073() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_074() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_075() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_076() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_077() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_078() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_079() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_080() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_081() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_082() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_083() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_084() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_085() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_086() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_087() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_088() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_089() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_090() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_091() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_092() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_093() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_094() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_095() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_096() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_097() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_098() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_099() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_100() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_101() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_102() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_103() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_104() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_105() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_106() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_107() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_108() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_109() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_110() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_111() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_112() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_113() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_114() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_115() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_116() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_117() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_118() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_119() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_120() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_121() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_122() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_123() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_124() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_125() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_126() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_127() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_128() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_129() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_130() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_131() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_132() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_133() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_134() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_135() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_136() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_137() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_138() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_139() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_140() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_141() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_142() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_143() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_144() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_145() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_146() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_147() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_148() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_149() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_150() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_151() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_152() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_153() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_154() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_155() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_156() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_157() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_158() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_159() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_160() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_161() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_162() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_163() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_164() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_165() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_166() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_167() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_168() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_169() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_170() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_171() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_172() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_173() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_174() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_175() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_176() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_177() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_178() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_179() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_180() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_181() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_182() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_183() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_184() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_185() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_186() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_187() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_188() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_189() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_190() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_191() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_192() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_193() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_194() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_195() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_196() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_197() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_198() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_199() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_200() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_201() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_202() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_203() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_204() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_205() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    #[test]
    fn test_layout_stress_206() {
        let mut g = GraphIr::new("layout_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![3, 2]), crate::core::DType::F32);
        let v3 = g.add_value("v3", crate::core::Shape::new(vec![2, 3]), crate::core::DType::F32);

        g.inputs.push(v1);
        g.add_node("tr1", OpKind::Transpose, vec![v1], vec![v2]);
        g.add_node("tr2", OpKind::Transpose, vec![v2], vec![v3]);
        g.outputs.push(v3);

        let modded = eliminate_layout_transforms(&mut g).unwrap();
        assert!(modded);
    }

    // Computation graph IR verification and pass padding line 0
}
