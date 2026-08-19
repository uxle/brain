//! # Execution Tape & Tracing Registry
//!
//! Captures dynamic execution graphs for debugging, graph visualization,
//! and static graph compilation.

pub mod builder;
pub mod node;
pub mod fused;
pub mod prune;

pub use builder::TapeBuilder;
pub use node::OpRecord;
pub use fused::TapeFusionPass;
pub use prune::TapePruner;

use std::cell::RefCell;
thread_local! {
    static ACTIVE_TAPE: RefCell<Option<Tape>> = const { RefCell::new(None) };
}

/// Execution tape capturing operation records.
#[derive(Debug, Clone, Default)]
pub struct Tape {
    records: Vec<OpRecord>,
}

impl Tape {
    /// Creates a new empty execution tape.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an operation.
    pub fn record(&mut self, record: OpRecord) {
        self.records.push(record);
    }

    /// Returns the number of recorded operations.
    pub fn op_count(&self) -> usize {
        self.records.len()
    }

    /// Returns an immutable slice of all records.
    pub fn records(&self) -> &[OpRecord] {
        &self.records
    }

    /// Clears the tape.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Drains all records from the tape.
    pub fn drain(&mut self) -> std::vec::Drain<'_, OpRecord> {
        self.records.drain(..)
    }

    /// Resets the tape and shrinks capacity to release memory.
    pub fn reset(&mut self) {
        self.records.clear();
        self.records.shrink_to_fit();
    }
}

/// Starts recording on the current thread.
pub fn start_recording() {
    ACTIVE_TAPE.with(|t| {
        *t.borrow_mut() = Some(Tape::new());
    });
}

/// Stops recording on the current thread and returns the recorded tape.
pub fn stop_recording() -> Option<Tape> {
    ACTIVE_TAPE.with(|t| t.borrow_mut().take())
}

/// Runs a closure with active tape recording enabled.
pub fn with_tape<F, R>(f: F) -> (R, Tape)
where
    F: FnOnce() -> R,
{
    start_recording();
    let res = f();
    let tape = stop_recording().unwrap_or_default();
    (res, tape)
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
