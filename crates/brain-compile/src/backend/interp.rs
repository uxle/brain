//! # Reference Interpreter Backend
//!
//! Register-based reference execution engine for `IrGraph` evaluation.

use crate::core::CompilationError;
use crate::ir::{IrGraph, OpKind};
use brain_core::Tensor;
use std::collections::HashMap;

/// Reference interpreter for direct graph execution.
#[derive(Default)]
pub struct Interpreter;

impl Interpreter {
    /// Creates a new `Interpreter`.
    pub fn new() -> Self {
        Self
    }

    /// Returns backend name.
    pub fn name(&self) -> &str {
        "interpreter"
    }

    /// Evaluates the graph given input tensors and returns output tensors.
    pub fn evaluate(
        &self,
        graph: &IrGraph,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>, CompilationError> {
        let mut registers: HashMap<usize, Tensor> = HashMap::new();

        for (i, &in_id) in graph.inputs.iter().enumerate() {
            if i < inputs.len() {
                registers.insert(in_id, inputs[i].clone());
            }
        }

        for node in &graph.nodes {
            match &node.kind {
                OpKind::Constant(c) => {
                    let _out_val = &graph.values[node.output];
                    registers.insert(node.output, Tensor::scalar(*c));
                }
                OpKind::Add if node.inputs.len() == 2 => {
                    if let (Some(a), Some(b)) = (
                        registers.get(&node.inputs[0]),
                        registers.get(&node.inputs[1]),
                    ) {
                        registers.insert(node.output, a + b);
                    }
                }
                OpKind::Mul if node.inputs.len() == 2 => {
                    if let (Some(a), Some(b)) = (
                        registers.get(&node.inputs[0]),
                        registers.get(&node.inputs[1]),
                    ) {
                        registers.insert(node.output, a * b);
                    }
                }
                OpKind::Sub if node.inputs.len() == 2 => {
                    if let (Some(a), Some(b)) = (
                        registers.get(&node.inputs[0]),
                        registers.get(&node.inputs[1]),
                    ) {
                        registers.insert(node.output, a - b);
                    }
                }
                OpKind::Div if node.inputs.len() == 2 => {
                    if let (Some(a), Some(b)) = (
                        registers.get(&node.inputs[0]),
                        registers.get(&node.inputs[1]),
                    ) {
                        registers.insert(node.output, a / b);
                    }
                }
                OpKind::Relu if !node.inputs.is_empty() => {
                    if let Some(a) = registers.get(&node.inputs[0]) {
                        registers.insert(node.output, a.map(|v| v.max(0.0)));
                    }
                }
                _ => {}
            }
        }

        let mut outputs = Vec::new();
        for &out_id in &graph.outputs {
            if let Some(t) = registers.get(&out_id) {
                outputs.push(t.clone());
            }
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
