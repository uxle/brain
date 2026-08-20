//! # Tape Record Nodes
//!
//! Strongly-typed records representing operation executions on the autograd tape.

/// An operation record stored on the execution tape.
#[derive(Debug, Clone, PartialEq)]
pub struct OpRecord {
    /// Name of the executed operation.
    pub op_name: String,
    /// IDs of input nodes.
    pub inputs: Vec<usize>,
    /// IDs of output nodes.
    pub outputs: Vec<usize>,
    /// Output tensor shapes.
    pub shapes: Vec<Vec<usize>>,
}

impl OpRecord {
    /// Creates a new op record.
    pub fn new(
        op_name: impl Into<String>,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
        shapes: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            op_name: op_name.into(),
            inputs,
            outputs,
            shapes,
        }
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
