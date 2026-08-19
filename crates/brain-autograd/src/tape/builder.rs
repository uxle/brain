//! # Fluent Tape Graph Builder
//!
//! Provides a programmatic builder API for constructing and tracing static execution tapes.

use crate::tape::{OpRecord, Tape};

/// Programmatic builder for constructing execution tapes.
#[derive(Debug, Default)]
pub struct TapeBuilder {
    tape: Tape,
}

impl TapeBuilder {
    /// Creates a new `TapeBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an operation record.
    pub fn add_op(mut self, op: OpRecord) -> Self {
        self.tape.record(op);
        self
    }

    /// Builds and returns the resulting tape.
    pub fn build(self) -> Tape {
        self.tape
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
}
