//! # Export Intermediate Representation (IR)
//!
//! Format-neutral computational graph containing topological node sequences and constant bindings.

/// Single computational node in the intermediate graph.
#[derive(Debug, Clone)]
pub struct ExportNode {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// Format-neutral intermediate graph for export.
#[derive(Debug, Clone)]
pub struct ExportIr {
    pub name: String,
    pub nodes: Vec<ExportNode>,
}

impl ExportIr {
    /// Creates a new `ExportIr`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
        }
    }

    /// Adds a node to the intermediate graph.
    pub fn add_node(&mut self, node: ExportNode) {
        self.nodes.push(node);
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
