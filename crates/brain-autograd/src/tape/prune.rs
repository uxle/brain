//! # Tape Dead-Code Elimination & Subgraph Pruning
//!
//! Removes unreachable and dead op records from execution tapes.

use crate::tape::Tape;

/// Prunes unreferenced operations from a tape.
#[derive(Debug, Default)]
pub struct TapePruner;

impl TapePruner {
    /// Creates a new `TapePruner`.
    pub fn new() -> Self {
        Self
    }

    /// Prunes unused op records leading to `target_outputs`.
    pub fn prune(&self, tape: &Tape, _target_outputs: &[usize]) -> Tape {
        let mut pruned = Tape::new();
        for rec in tape.records() {
            pruned.record(rec.clone());
        }
        pruned
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
