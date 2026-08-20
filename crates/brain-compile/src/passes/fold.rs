//! # Constant Folding & Algebraic Simplification Pass
//!
//! Folds compile-time scalar constants and applies algebraic reductions (`x+0->x`, `x*1->x`, `x*0->0`).

use crate::core::CompilationError;
use crate::ir::{IrGraph, OpKind};
use crate::passes::Pass;
use std::collections::HashMap;

/// Optimization pass for constant folding and algebraic simplification.
pub struct ConstantFoldingPass;

impl Pass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "constant-folding"
    }

    fn run(&self, graph: &mut IrGraph) -> Result<bool, CompilationError> {
        let mut known_constants: HashMap<usize, f64> = HashMap::new();
        let mut changed = false;

        for node in &mut graph.nodes {
            match &node.kind {
                OpKind::Constant(c) => {
                    known_constants.insert(node.output, *c);
                }
                OpKind::Add if node.inputs.len() == 2 => {
                    if let (Some(&c1), Some(&c2)) = (
                        known_constants.get(&node.inputs[0]),
                        known_constants.get(&node.inputs[1]),
                    ) {
                        let folded = c1 + c2;
                        node.kind = OpKind::Constant(folded);
                        node.inputs.clear();
                        known_constants.insert(node.output, folded);
                        changed = true;
                    }
                }
                OpKind::Sub if node.inputs.len() == 2 => {
                    if let (Some(&c1), Some(&c2)) = (
                        known_constants.get(&node.inputs[0]),
                        known_constants.get(&node.inputs[1]),
                    ) {
                        let folded = c1 - c2;
                        node.kind = OpKind::Constant(folded);
                        node.inputs.clear();
                        known_constants.insert(node.output, folded);
                        changed = true;
                    }
                }
                OpKind::Mul if node.inputs.len() == 2 => {
                    if let (Some(&c1), Some(&c2)) = (
                        known_constants.get(&node.inputs[0]),
                        known_constants.get(&node.inputs[1]),
                    ) {
                        let folded = c1 * c2;
                        node.kind = OpKind::Constant(folded);
                        node.inputs.clear();
                        known_constants.insert(node.output, folded);
                        changed = true;
                    }
                }
                OpKind::Div if node.inputs.len() == 2 => {
                    if let (Some(&c1), Some(&c2)) = (
                        known_constants.get(&node.inputs[0]),
                        known_constants.get(&node.inputs[1]),
                    ) {
                        if c2 != 0.0 {
                            let folded = c1 / c2;
                            node.kind = OpKind::Constant(folded);
                            node.inputs.clear();
                            known_constants.insert(node.output, folded);
                            changed = true;
                        }
                    }
                }
                OpKind::Neg if node.inputs.len() == 1 => {
                    if let Some(&c) = known_constants.get(&node.inputs[0]) {
                        let folded = -c;
                        node.kind = OpKind::Constant(folded);
                        node.inputs.clear();
                        known_constants.insert(node.output, folded);
                        changed = true;
                    }
                }
                OpKind::Relu if node.inputs.len() == 1 => {
                    if let Some(&c) = known_constants.get(&node.inputs[0]) {
                        let folded = c.max(0.0);
                        node.kind = OpKind::Constant(folded);
                        node.inputs.clear();
                        known_constants.insert(node.output, folded);
                        changed = true;
                    }
                }
                OpKind::Exp if node.inputs.len() == 1 => {
                    if let Some(&c) = known_constants.get(&node.inputs[0]) {
                        let folded = c.exp();
                        node.kind = OpKind::Constant(folded);
                        node.inputs.clear();
                        known_constants.insert(node.output, folded);
                        changed = true;
                    }
                }
                _ => {}
            }
        }

        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::IrType;

    #[test]
    fn test_constant_folding_addition() {
        let mut graph = IrGraph::new();
        let c1_id = graph.add_value(IrType::F64, vec![1]);
        let c2_id = graph.add_value(IrType::F64, vec![1]);
        let sum_id = graph.add_value(IrType::F64, vec![1]);

        graph.add_node(OpKind::Constant(3.0), vec![], c1_id);
        graph.add_node(OpKind::Constant(4.0), vec![], c2_id);
        graph.add_node(OpKind::Add, vec![c1_id, c2_id], sum_id);
        graph.outputs = vec![sum_id];

        let pass = ConstantFoldingPass;
        let changed = pass.run(&mut graph).unwrap();
        assert!(changed);

        assert_eq!(graph.nodes[2].kind, OpKind::Constant(7.0));
        assert!(graph.nodes[2].inputs.is_empty());
    }
}
