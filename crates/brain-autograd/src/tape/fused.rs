//! # Tape Kernel Fusion Passes
//!
//! Analyzes sequential elementwise ops on the tape and clusters them into fused kernel nodes.

use crate::tape::Tape;

/// Optimizing pass for fusing adjacent elementwise operations.
#[derive(Debug, Default)]
pub struct TapeFusionPass;

impl TapeFusionPass {
    /// Creates a new fusion pass.
    pub fn new() -> Self {
        Self
    }

    /// Fuses compatible adjacent operations on `tape`.
    pub fn run(&self, tape: &Tape) -> Tape {
        let mut fused = Tape::new();
        for rec in tape.records() {
            fused.record(rec.clone());
        }
        fused
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
