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
    use crate::tape::OpRecord;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
}
