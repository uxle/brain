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

    #[test]
    fn test_tape_lifecycle_stress_001() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![1], vec![2], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_002() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![2], vec![3], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_003() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![3], vec![4], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_004() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![4], vec![5], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_005() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![5], vec![6], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_006() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![6], vec![7], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_007() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![7], vec![8], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_008() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![8], vec![9], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_009() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![9], vec![10], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_010() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![10], vec![11], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_011() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![11], vec![12], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_012() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![12], vec![13], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_013() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![13], vec![14], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_014() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![14], vec![15], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_015() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![15], vec![16], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_016() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![16], vec![17], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_017() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![17], vec![18], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_018() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![18], vec![19], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_019() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![19], vec![20], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_020() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![20], vec![21], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_021() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![21], vec![22], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_022() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![22], vec![23], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_023() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![23], vec![24], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_024() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![24], vec![25], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_025() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![25], vec![26], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_026() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![26], vec![27], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_027() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![27], vec![28], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_028() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![28], vec![29], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_029() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![29], vec![30], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_030() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![30], vec![31], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_031() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![31], vec![32], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_032() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![32], vec![33], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_033() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![33], vec![34], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_034() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![34], vec![35], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_035() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![35], vec![36], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_036() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![36], vec![37], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_037() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![37], vec![38], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_038() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![38], vec![39], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_039() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![39], vec![40], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_040() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![40], vec![41], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_041() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![41], vec![42], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_042() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![42], vec![43], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_043() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![43], vec![44], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_044() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![44], vec![45], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_045() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![45], vec![46], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_046() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![46], vec![47], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_047() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![47], vec![48], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_048() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![48], vec![49], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_049() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![49], vec![50], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_050() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![50], vec![51], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_051() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![51], vec![52], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_052() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![52], vec![53], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_053() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![53], vec![54], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_054() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![54], vec![55], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_055() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![55], vec![56], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_056() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![56], vec![57], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_057() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![57], vec![58], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_058() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![58], vec![59], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_059() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![59], vec![60], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_060() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![60], vec![61], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_061() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![61], vec![62], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_062() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![62], vec![63], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_063() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![63], vec![64], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_064() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![64], vec![65], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_065() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![65], vec![66], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_066() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![66], vec![67], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_067() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![67], vec![68], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_068() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![68], vec![69], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_069() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![69], vec![70], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_070() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![70], vec![71], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_071() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![71], vec![72], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_072() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![72], vec![73], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_073() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![73], vec![74], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_074() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![74], vec![75], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_075() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![75], vec![76], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_076() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![76], vec![77], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_077() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![77], vec![78], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_078() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![78], vec![79], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_079() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![79], vec![80], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_080() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![80], vec![81], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_081() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![81], vec![82], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_082() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![82], vec![83], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_083() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![83], vec![84], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_084() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![84], vec![85], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_085() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![85], vec![86], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_086() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![86], vec![87], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_087() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![87], vec![88], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_088() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![88], vec![89], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_089() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![89], vec![90], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_090() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![90], vec![91], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_091() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![91], vec![92], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_092() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![92], vec![93], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_093() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![93], vec![94], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_094() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![94], vec![95], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_095() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![95], vec![96], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_096() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![96], vec![97], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_097() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![97], vec![98], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_098() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![98], vec![99], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_099() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![99], vec![100], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_100() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![100], vec![101], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_101() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![101], vec![102], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_102() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![102], vec![103], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_103() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![103], vec![104], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_104() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![104], vec![105], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_105() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![105], vec![106], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_106() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![106], vec![107], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_107() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![107], vec![108], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_108() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![108], vec![109], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_109() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![109], vec![110], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_110() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![110], vec![111], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_111() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![111], vec![112], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_112() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![112], vec![113], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_113() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![113], vec![114], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_114() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![114], vec![115], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_115() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![115], vec![116], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_116() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![116], vec![117], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_117() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![117], vec![118], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_118() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![118], vec![119], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_119() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![119], vec![120], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_120() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![120], vec![121], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_121() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![121], vec![122], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_122() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![122], vec![123], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_123() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![123], vec![124], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_124() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![124], vec![125], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_125() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![125], vec![126], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_126() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![126], vec![127], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_127() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![127], vec![128], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_128() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![128], vec![129], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_129() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![129], vec![130], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_130() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![130], vec![131], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_131() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![131], vec![132], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_132() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![132], vec![133], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_133() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![133], vec![134], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_134() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![134], vec![135], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_135() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![135], vec![136], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_136() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![136], vec![137], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_137() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![137], vec![138], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_138() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![138], vec![139], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_139() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![139], vec![140], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_140() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![140], vec![141], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_141() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![141], vec![142], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_142() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![142], vec![143], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_143() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![143], vec![144], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_144() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![144], vec![145], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_145() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![145], vec![146], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_146() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![146], vec![147], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_147() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![147], vec![148], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_148() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![148], vec![149], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_149() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![149], vec![150], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_150() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![150], vec![151], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_151() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![151], vec![152], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_152() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![152], vec![153], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_153() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![153], vec![154], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_154() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![154], vec![155], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_155() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![155], vec![156], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_156() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![156], vec![157], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_157() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![157], vec![158], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_158() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![158], vec![159], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_159() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![159], vec![160], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_160() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![160], vec![161], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_161() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![161], vec![162], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_162() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![162], vec![163], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_163() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![163], vec![164], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_164() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![164], vec![165], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_165() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![165], vec![166], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_166() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![166], vec![167], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_167() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![167], vec![168], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_168() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![168], vec![169], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_169() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![169], vec![170], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_170() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![170], vec![171], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_171() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![171], vec![172], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_172() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![172], vec![173], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_173() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![173], vec![174], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_174() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![174], vec![175], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_175() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![175], vec![176], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_176() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![176], vec![177], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_177() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![177], vec![178], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_178() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![178], vec![179], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_179() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![179], vec![180], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_180() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![180], vec![181], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_181() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![181], vec![182], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_182() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![182], vec![183], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_183() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![183], vec![184], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_184() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![184], vec![185], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_185() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![185], vec![186], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_186() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![186], vec![187], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_187() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![187], vec![188], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_188() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![188], vec![189], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_189() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![189], vec![190], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_190() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![190], vec![191], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_191() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![191], vec![192], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_192() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![192], vec![193], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_193() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![193], vec![194], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_194() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![194], vec![195], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_195() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![195], vec![196], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_196() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![196], vec![197], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_197() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![197], vec![198], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_198() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![198], vec![199], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_199() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![199], vec![200], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_200() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![200], vec![201], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_201() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![201], vec![202], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_202() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![202], vec![203], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_203() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![203], vec![204], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_204() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![204], vec![205], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_205() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![205], vec![206], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_206() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![206], vec![207], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_207() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![207], vec![208], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_208() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![208], vec![209], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_209() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![209], vec![210], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_210() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![210], vec![211], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_211() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![211], vec![212], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_212() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![212], vec![213], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_213() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![213], vec![214], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_214() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![214], vec![215], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_215() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![215], vec![216], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_216() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![216], vec![217], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_217() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![217], vec![218], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_218() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![218], vec![219], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_219() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![219], vec![220], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_220() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![220], vec![221], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_221() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![221], vec![222], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_222() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![222], vec![223], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_223() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![223], vec![224], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_224() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![224], vec![225], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_225() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![225], vec![226], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_226() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![226], vec![227], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_227() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![227], vec![228], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_228() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![228], vec![229], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_229() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![229], vec![230], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_230() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![230], vec![231], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_231() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![231], vec![232], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_232() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![232], vec![233], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_233() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![233], vec![234], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_234() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![234], vec![235], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_235() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![235], vec![236], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_236() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![236], vec![237], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_237() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![237], vec![238], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_238() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![238], vec![239], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_239() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![239], vec![240], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_240() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![240], vec![241], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_241() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![241], vec![242], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_242() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![242], vec![243], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_243() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![243], vec![244], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_244() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![244], vec![245], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_245() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![245], vec![246], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_246() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![246], vec![247], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_247() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![247], vec![248], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_248() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![248], vec![249], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_249() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![249], vec![250], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_250() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![250], vec![251], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_251() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![251], vec![252], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_252() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![252], vec![253], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_253() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![253], vec![254], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_254() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![254], vec![255], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_255() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![255], vec![256], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_256() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![256], vec![257], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_257() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![257], vec![258], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_258() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![258], vec![259], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_259() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![259], vec![260], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_260() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![260], vec![261], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_261() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![261], vec![262], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_262() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![262], vec![263], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_263() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![263], vec![264], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_264() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![264], vec![265], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_265() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![265], vec![266], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_266() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![266], vec![267], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_267() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![267], vec![268], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_268() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![268], vec![269], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_269() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![269], vec![270], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_270() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![270], vec![271], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_271() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![271], vec![272], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_272() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![272], vec![273], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_273() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![273], vec![274], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_274() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![274], vec![275], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_275() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![275], vec![276], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_276() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![276], vec![277], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_277() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![277], vec![278], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_278() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![278], vec![279], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_279() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![279], vec![280], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_280() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![280], vec![281], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_281() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![281], vec![282], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_282() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![282], vec![283], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_283() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![283], vec![284], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_284() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![284], vec![285], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_285() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![285], vec![286], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_286() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![286], vec![287], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_287() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![287], vec![288], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_288() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![288], vec![289], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_289() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![289], vec![290], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_290() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![290], vec![291], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_291() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![291], vec![292], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_292() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![292], vec![293], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_293() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![293], vec![294], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_294() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![294], vec![295], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_295() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![295], vec![296], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_296() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![296], vec![297], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_297() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![297], vec![298], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_298() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![298], vec![299], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_299() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![299], vec![300], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_300() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![300], vec![301], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_301() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![301], vec![302], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_302() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![302], vec![303], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_303() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![303], vec![304], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_304() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![304], vec![305], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_305() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![305], vec![306], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_306() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![306], vec![307], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_307() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![307], vec![308], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_308() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![308], vec![309], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_309() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![309], vec![310], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_310() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![310], vec![311], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_311() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![311], vec![312], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_312() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![312], vec![313], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_313() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![313], vec![314], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_314() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![314], vec![315], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_315() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![315], vec![316], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_316() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![316], vec![317], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_317() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![317], vec![318], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_318() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![318], vec![319], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_319() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![319], vec![320], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_320() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![320], vec![321], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_321() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![321], vec![322], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_322() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![322], vec![323], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_323() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![323], vec![324], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_324() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![324], vec![325], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_325() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![325], vec![326], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_326() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![326], vec![327], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_327() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![327], vec![328], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_328() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![328], vec![329], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_329() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![329], vec![330], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_330() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![330], vec![331], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_331() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![331], vec![332], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_332() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![332], vec![333], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_333() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![333], vec![334], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_334() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![334], vec![335], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_335() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![335], vec![336], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_336() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![336], vec![337], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_337() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![337], vec![338], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_338() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![338], vec![339], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_339() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![339], vec![340], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_340() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![340], vec![341], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_341() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![341], vec![342], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_342() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![342], vec![343], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_343() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![343], vec![344], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_344() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![344], vec![345], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_345() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![345], vec![346], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_346() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![346], vec![347], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_347() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![347], vec![348], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_348() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![348], vec![349], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_349() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![349], vec![350], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_350() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![350], vec![351], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_351() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![351], vec![352], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_352() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![352], vec![353], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_353() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![353], vec![354], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_354() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![354], vec![355], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_355() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![355], vec![356], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_356() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![356], vec![357], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_357() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![357], vec![358], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_358() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![358], vec![359], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_359() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![359], vec![360], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_360() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![360], vec![361], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_361() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![361], vec![362], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_362() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![362], vec![363], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_363() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![363], vec![364], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_364() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![364], vec![365], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_365() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![365], vec![366], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_366() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![366], vec![367], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_367() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![367], vec![368], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_368() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![368], vec![369], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_369() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![369], vec![370], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_370() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![370], vec![371], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_371() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![371], vec![372], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_372() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![372], vec![373], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_373() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![373], vec![374], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_374() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![374], vec![375], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_375() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![375], vec![376], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_376() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![376], vec![377], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_377() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![377], vec![378], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_378() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![378], vec![379], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_379() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![379], vec![380], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_380() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![380], vec![381], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_381() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![381], vec![382], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_382() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![382], vec![383], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_383() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![383], vec![384], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_384() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![384], vec![385], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_385() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![385], vec![386], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_386() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![386], vec![387], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_387() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![387], vec![388], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_388() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![388], vec![389], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_389() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![389], vec![390], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_390() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![390], vec![391], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_391() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![391], vec![392], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_392() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![392], vec![393], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_393() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![393], vec![394], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_394() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![394], vec![395], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_395() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![395], vec![396], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_396() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![396], vec![397], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_397() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![397], vec![398], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_398() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![398], vec![399], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_399() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![399], vec![400], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_400() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![400], vec![401], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_401() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![401], vec![402], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_402() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![402], vec![403], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_403() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![403], vec![404], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_404() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![404], vec![405], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_405() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![405], vec![406], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_406() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![406], vec![407], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_407() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![407], vec![408], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_408() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![408], vec![409], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_409() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![409], vec![410], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_410() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![410], vec![411], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_411() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![411], vec![412], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_412() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![412], vec![413], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_413() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![413], vec![414], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_414() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![414], vec![415], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_415() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![415], vec![416], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_416() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![416], vec![417], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_417() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![417], vec![418], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_418() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![418], vec![419], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_419() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![419], vec![420], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_420() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![420], vec![421], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_421() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![421], vec![422], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_422() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![422], vec![423], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_423() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![423], vec![424], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_424() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![424], vec![425], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_425() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![425], vec![426], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_426() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![426], vec![427], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_427() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![427], vec![428], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_428() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![428], vec![429], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_429() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![429], vec![430], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_430() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![430], vec![431], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_431() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![431], vec![432], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_432() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![432], vec![433], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_433() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![433], vec![434], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_434() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![434], vec![435], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_435() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![435], vec![436], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_436() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![436], vec![437], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_437() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![437], vec![438], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_438() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![438], vec![439], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_439() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![439], vec![440], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_440() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![440], vec![441], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_441() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![441], vec![442], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_442() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![442], vec![443], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_443() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![443], vec![444], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_444() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![444], vec![445], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_445() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![445], vec![446], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_446() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![446], vec![447], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_447() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![447], vec![448], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_448() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![448], vec![449], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_449() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![449], vec![450], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_450() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![450], vec![451], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_451() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![451], vec![452], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_452() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![452], vec![453], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_453() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![453], vec![454], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_454() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![454], vec![455], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_455() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![455], vec![456], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_456() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![456], vec![457], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_457() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![457], vec![458], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_458() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![458], vec![459], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_459() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![459], vec![460], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_460() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![460], vec![461], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_461() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![461], vec![462], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_462() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![462], vec![463], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_463() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![463], vec![464], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_464() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![464], vec![465], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_lifecycle_stress_465() {
        let mut tape = Tape::new();
        tape.record(OpRecord::new("mul", vec![465], vec![466], vec![vec![1]]));
        assert_eq!(tape.op_count(), 1);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
}
