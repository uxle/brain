//! # Process Groups & Communicators
//!
//! Subgroup management for hybrid data, pipeline, and tensor parallelism partitioning.

/// Process group managing a subset of cluster ranks.
#[derive(Debug, Clone)]
pub struct ProcessGroup {
    pub name: String,
    pub ranks: Vec<usize>,
}

impl ProcessGroup {
    /// Creates a new `ProcessGroup`.
    pub fn new(name: impl Into<String>, ranks: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            ranks,
        }
    }

    /// Returns the number of processes in the group.
    pub fn size(&self) -> usize {
        self.ranks.len()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
