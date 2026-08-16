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

    #[test]
    fn test_tape_builder_stress_001() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![1], vec![2], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_002() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![2], vec![3], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_003() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![3], vec![4], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_004() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![4], vec![5], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_005() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![5], vec![6], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_006() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![6], vec![7], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_007() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![7], vec![8], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_008() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![8], vec![9], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_009() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![9], vec![10], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_010() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![10], vec![11], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_011() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![11], vec![12], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_012() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![12], vec![13], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_013() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![13], vec![14], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_014() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![14], vec![15], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_015() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![15], vec![16], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_016() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![16], vec![17], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_017() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![17], vec![18], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_018() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![18], vec![19], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_019() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![19], vec![20], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_020() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![20], vec![21], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_021() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![21], vec![22], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_022() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![22], vec![23], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_023() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![23], vec![24], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_024() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![24], vec![25], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_025() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![25], vec![26], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_026() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![26], vec![27], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_027() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![27], vec![28], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_028() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![28], vec![29], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_029() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![29], vec![30], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_030() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![30], vec![31], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_031() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![31], vec![32], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_032() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![32], vec![33], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_033() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![33], vec![34], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_034() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![34], vec![35], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_035() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![35], vec![36], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_036() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![36], vec![37], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_037() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![37], vec![38], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_038() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![38], vec![39], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_039() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![39], vec![40], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_040() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![40], vec![41], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_041() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![41], vec![42], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_042() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![42], vec![43], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_043() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![43], vec![44], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_044() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![44], vec![45], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_045() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![45], vec![46], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_046() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![46], vec![47], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_047() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![47], vec![48], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_048() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![48], vec![49], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_049() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![49], vec![50], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_050() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![50], vec![51], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_051() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![51], vec![52], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_052() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![52], vec![53], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_053() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![53], vec![54], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_054() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![54], vec![55], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_055() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![55], vec![56], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_056() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![56], vec![57], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_057() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![57], vec![58], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_058() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![58], vec![59], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_059() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![59], vec![60], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_060() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![60], vec![61], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_061() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![61], vec![62], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_062() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![62], vec![63], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_063() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![63], vec![64], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_064() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![64], vec![65], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_065() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![65], vec![66], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_066() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![66], vec![67], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_067() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![67], vec![68], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_068() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![68], vec![69], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_069() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![69], vec![70], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_070() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![70], vec![71], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_071() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![71], vec![72], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_072() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![72], vec![73], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_073() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![73], vec![74], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_074() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![74], vec![75], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_075() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![75], vec![76], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_076() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![76], vec![77], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_077() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![77], vec![78], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_078() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![78], vec![79], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_079() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![79], vec![80], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_080() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![80], vec![81], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_081() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![81], vec![82], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_082() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![82], vec![83], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_083() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![83], vec![84], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_084() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![84], vec![85], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_085() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![85], vec![86], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_086() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![86], vec![87], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_087() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![87], vec![88], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_088() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![88], vec![89], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_089() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![89], vec![90], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_090() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![90], vec![91], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_091() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![91], vec![92], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_092() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![92], vec![93], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_093() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![93], vec![94], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_094() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![94], vec![95], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_095() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![95], vec![96], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_096() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![96], vec![97], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_097() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![97], vec![98], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_098() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![98], vec![99], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_099() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![99], vec![100], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_100() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![100], vec![101], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_101() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![101], vec![102], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_102() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![102], vec![103], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_103() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![103], vec![104], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_104() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![104], vec![105], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_105() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![105], vec![106], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_106() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![106], vec![107], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_107() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![107], vec![108], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_108() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![108], vec![109], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_109() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![109], vec![110], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_110() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![110], vec![111], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_111() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![111], vec![112], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_112() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![112], vec![113], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_113() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![113], vec![114], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_114() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![114], vec![115], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_115() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![115], vec![116], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_116() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![116], vec![117], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_117() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![117], vec![118], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_118() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![118], vec![119], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_119() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![119], vec![120], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_120() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![120], vec![121], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_121() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![121], vec![122], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_122() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![122], vec![123], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_123() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![123], vec![124], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_124() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![124], vec![125], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_125() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![125], vec![126], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_126() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![126], vec![127], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_127() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![127], vec![128], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_128() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![128], vec![129], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_129() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![129], vec![130], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_130() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![130], vec![131], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_131() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![131], vec![132], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_132() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![132], vec![133], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_133() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![133], vec![134], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_134() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![134], vec![135], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_135() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![135], vec![136], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_136() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![136], vec![137], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_137() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![137], vec![138], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_138() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![138], vec![139], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_139() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![139], vec![140], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_140() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![140], vec![141], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_141() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![141], vec![142], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_142() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![142], vec![143], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_143() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![143], vec![144], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_144() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![144], vec![145], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_145() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![145], vec![146], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_146() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![146], vec![147], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_147() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![147], vec![148], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_148() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![148], vec![149], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_149() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![149], vec![150], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_150() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![150], vec![151], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_151() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![151], vec![152], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_152() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![152], vec![153], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_153() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![153], vec![154], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_154() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![154], vec![155], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_155() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![155], vec![156], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_156() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![156], vec![157], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_157() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![157], vec![158], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_158() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![158], vec![159], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_159() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![159], vec![160], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_160() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![160], vec![161], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_161() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![161], vec![162], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_162() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![162], vec![163], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_163() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![163], vec![164], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_164() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![164], vec![165], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_165() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![165], vec![166], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_166() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![166], vec![167], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_167() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![167], vec![168], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_168() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![168], vec![169], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_169() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![169], vec![170], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_170() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![170], vec![171], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_171() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![171], vec![172], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_172() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![172], vec![173], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_173() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![173], vec![174], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_174() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![174], vec![175], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_175() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![175], vec![176], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_176() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![176], vec![177], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_177() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![177], vec![178], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_178() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![178], vec![179], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_179() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![179], vec![180], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_180() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![180], vec![181], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_181() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![181], vec![182], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_182() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![182], vec![183], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_183() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![183], vec![184], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_184() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![184], vec![185], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_185() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![185], vec![186], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_186() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![186], vec![187], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_187() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![187], vec![188], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_188() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![188], vec![189], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_189() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![189], vec![190], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_190() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![190], vec![191], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_191() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![191], vec![192], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_192() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![192], vec![193], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_193() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![193], vec![194], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_194() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![194], vec![195], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_195() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![195], vec![196], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_196() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![196], vec![197], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_197() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![197], vec![198], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_198() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![198], vec![199], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_199() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![199], vec![200], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_200() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![200], vec![201], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_201() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![201], vec![202], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_202() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![202], vec![203], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_203() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![203], vec![204], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_204() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![204], vec![205], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_205() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![205], vec![206], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_206() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![206], vec![207], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_207() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![207], vec![208], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_208() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![208], vec![209], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_209() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![209], vec![210], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_210() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![210], vec![211], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_211() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![211], vec![212], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_212() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![212], vec![213], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_213() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![213], vec![214], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_214() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![214], vec![215], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_215() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![215], vec![216], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_216() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![216], vec![217], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_217() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![217], vec![218], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_218() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![218], vec![219], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_219() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![219], vec![220], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_220() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![220], vec![221], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_221() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![221], vec![222], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_222() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![222], vec![223], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_223() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![223], vec![224], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_224() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![224], vec![225], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_225() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![225], vec![226], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_226() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![226], vec![227], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_227() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![227], vec![228], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_228() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![228], vec![229], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_229() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![229], vec![230], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_230() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![230], vec![231], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_231() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![231], vec![232], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_232() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![232], vec![233], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_233() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![233], vec![234], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_234() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![234], vec![235], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_235() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![235], vec![236], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_236() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![236], vec![237], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_237() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![237], vec![238], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_238() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![238], vec![239], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_239() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![239], vec![240], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_240() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![240], vec![241], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_241() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![241], vec![242], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_242() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![242], vec![243], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_243() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![243], vec![244], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_244() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![244], vec![245], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_245() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![245], vec![246], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_246() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![246], vec![247], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_247() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![247], vec![248], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_248() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![248], vec![249], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_249() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![249], vec![250], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_250() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![250], vec![251], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_251() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![251], vec![252], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_252() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![252], vec![253], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_253() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![253], vec![254], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_254() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![254], vec![255], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_255() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![255], vec![256], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_256() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![256], vec![257], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_257() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![257], vec![258], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_258() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![258], vec![259], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_259() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![259], vec![260], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_260() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![260], vec![261], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_261() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![261], vec![262], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_262() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![262], vec![263], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_263() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![263], vec![264], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_264() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![264], vec![265], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_265() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![265], vec![266], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_266() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![266], vec![267], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_267() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![267], vec![268], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_268() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![268], vec![269], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_269() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![269], vec![270], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_270() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![270], vec![271], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_271() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![271], vec![272], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_272() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![272], vec![273], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_273() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![273], vec![274], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_274() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![274], vec![275], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_275() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![275], vec![276], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_276() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![276], vec![277], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_277() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![277], vec![278], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_278() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![278], vec![279], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_279() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![279], vec![280], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_280() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![280], vec![281], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_281() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![281], vec![282], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_282() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![282], vec![283], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_283() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![283], vec![284], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_284() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![284], vec![285], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_285() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![285], vec![286], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_286() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![286], vec![287], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_287() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![287], vec![288], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_288() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![288], vec![289], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_289() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![289], vec![290], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_290() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![290], vec![291], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_291() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![291], vec![292], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_292() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![292], vec![293], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_293() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![293], vec![294], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_294() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![294], vec![295], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_295() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![295], vec![296], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_296() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![296], vec![297], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_297() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![297], vec![298], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_298() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![298], vec![299], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_299() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![299], vec![300], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_300() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![300], vec![301], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_301() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![301], vec![302], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_302() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![302], vec![303], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_303() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![303], vec![304], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_304() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![304], vec![305], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_305() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![305], vec![306], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_306() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![306], vec![307], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_307() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![307], vec![308], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_308() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![308], vec![309], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_309() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![309], vec![310], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_310() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![310], vec![311], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_311() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![311], vec![312], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_312() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![312], vec![313], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_313() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![313], vec![314], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_314() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![314], vec![315], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_315() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![315], vec![316], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_316() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![316], vec![317], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_317() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![317], vec![318], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_318() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![318], vec![319], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_319() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![319], vec![320], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_320() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![320], vec![321], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_321() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![321], vec![322], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_322() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![322], vec![323], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_323() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![323], vec![324], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_324() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![324], vec![325], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_325() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![325], vec![326], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_326() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![326], vec![327], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_327() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![327], vec![328], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_328() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![328], vec![329], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_329() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![329], vec![330], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_330() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![330], vec![331], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_331() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![331], vec![332], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_332() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![332], vec![333], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_333() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![333], vec![334], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_334() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![334], vec![335], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_335() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![335], vec![336], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_336() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![336], vec![337], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_337() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![337], vec![338], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_338() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![338], vec![339], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_339() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![339], vec![340], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_340() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![340], vec![341], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_341() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![341], vec![342], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_342() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![342], vec![343], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_343() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![343], vec![344], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_344() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![344], vec![345], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_345() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![345], vec![346], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_346() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![346], vec![347], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_347() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![347], vec![348], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_348() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![348], vec![349], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_349() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![349], vec![350], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_350() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![350], vec![351], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_351() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![351], vec![352], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_352() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![352], vec![353], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_353() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![353], vec![354], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_354() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![354], vec![355], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_355() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![355], vec![356], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_356() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![356], vec![357], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_357() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![357], vec![358], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_358() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![358], vec![359], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_359() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![359], vec![360], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_360() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![360], vec![361], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_361() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![361], vec![362], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_362() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![362], vec![363], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_363() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![363], vec![364], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_364() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![364], vec![365], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_365() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![365], vec![366], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_366() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![366], vec![367], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_367() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![367], vec![368], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_368() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![368], vec![369], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_369() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![369], vec![370], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_370() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![370], vec![371], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_371() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![371], vec![372], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_372() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![372], vec![373], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_373() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![373], vec![374], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_374() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![374], vec![375], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_375() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![375], vec![376], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_376() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![376], vec![377], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_377() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![377], vec![378], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_378() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![378], vec![379], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_379() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![379], vec![380], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_380() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![380], vec![381], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_381() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![381], vec![382], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_382() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![382], vec![383], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_383() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![383], vec![384], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_384() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![384], vec![385], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_385() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![385], vec![386], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_386() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![386], vec![387], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_387() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![387], vec![388], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_388() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![388], vec![389], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_389() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![389], vec![390], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_390() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![390], vec![391], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_391() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![391], vec![392], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_392() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![392], vec![393], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_393() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![393], vec![394], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_394() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![394], vec![395], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_395() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![395], vec![396], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_396() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![396], vec![397], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_397() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![397], vec![398], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_398() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![398], vec![399], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_399() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![399], vec![400], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_400() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![400], vec![401], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_401() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![401], vec![402], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_402() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![402], vec![403], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_403() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![403], vec![404], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_404() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![404], vec![405], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_405() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![405], vec![406], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_406() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![406], vec![407], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_407() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![407], vec![408], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_408() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![408], vec![409], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_409() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![409], vec![410], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_410() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![410], vec![411], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_411() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![411], vec![412], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_412() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![412], vec![413], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }

    #[test]
    fn test_tape_builder_stress_413() {
        let b = TapeBuilder::new()
            .add_op(OpRecord::new("exp", vec![413], vec![414], vec![vec![1]]));
        let tape = b.build();
        assert_eq!(tape.op_count(), 1);
    }
}
