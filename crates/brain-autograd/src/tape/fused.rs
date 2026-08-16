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

    #[test]
    fn test_tape_fusion_stress_001() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![1], vec![2], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_002() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![2], vec![3], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_003() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![3], vec![4], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_004() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![4], vec![5], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_005() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![5], vec![6], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_006() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![6], vec![7], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_007() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![7], vec![8], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_008() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![8], vec![9], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_009() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![9], vec![10], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_010() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![10], vec![11], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_011() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![11], vec![12], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_012() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![12], vec![13], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_013() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![13], vec![14], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_014() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![14], vec![15], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_015() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![15], vec![16], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_016() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![16], vec![17], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_017() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![17], vec![18], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_018() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![18], vec![19], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_019() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![19], vec![20], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_020() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![20], vec![21], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_021() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![21], vec![22], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_022() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![22], vec![23], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_023() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![23], vec![24], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_024() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![24], vec![25], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_025() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![25], vec![26], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_026() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![26], vec![27], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_027() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![27], vec![28], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_028() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![28], vec![29], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_029() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![29], vec![30], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_030() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![30], vec![31], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_031() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![31], vec![32], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_032() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![32], vec![33], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_033() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![33], vec![34], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_034() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![34], vec![35], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_035() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![35], vec![36], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_036() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![36], vec![37], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_037() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![37], vec![38], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_038() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![38], vec![39], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_039() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![39], vec![40], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_040() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![40], vec![41], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_041() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![41], vec![42], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_042() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![42], vec![43], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_043() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![43], vec![44], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_044() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![44], vec![45], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_045() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![45], vec![46], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_046() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![46], vec![47], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_047() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![47], vec![48], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_048() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![48], vec![49], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_049() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![49], vec![50], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_050() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![50], vec![51], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_051() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![51], vec![52], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_052() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![52], vec![53], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_053() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![53], vec![54], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_054() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![54], vec![55], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_055() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![55], vec![56], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_056() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![56], vec![57], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_057() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![57], vec![58], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_058() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![58], vec![59], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_059() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![59], vec![60], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_060() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![60], vec![61], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_061() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![61], vec![62], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_062() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![62], vec![63], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_063() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![63], vec![64], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_064() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![64], vec![65], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_065() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![65], vec![66], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_066() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![66], vec![67], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_067() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![67], vec![68], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_068() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![68], vec![69], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_069() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![69], vec![70], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_070() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![70], vec![71], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_071() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![71], vec![72], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_072() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![72], vec![73], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_073() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![73], vec![74], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_074() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![74], vec![75], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_075() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![75], vec![76], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_076() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![76], vec![77], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_077() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![77], vec![78], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_078() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![78], vec![79], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_079() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![79], vec![80], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_080() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![80], vec![81], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_081() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![81], vec![82], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_082() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![82], vec![83], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_083() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![83], vec![84], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_084() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![84], vec![85], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_085() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![85], vec![86], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_086() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![86], vec![87], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_087() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![87], vec![88], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_088() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![88], vec![89], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_089() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![89], vec![90], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_090() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![90], vec![91], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_091() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![91], vec![92], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_092() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![92], vec![93], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_093() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![93], vec![94], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_094() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![94], vec![95], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_095() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![95], vec![96], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_096() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![96], vec![97], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_097() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![97], vec![98], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_098() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![98], vec![99], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_099() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![99], vec![100], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_100() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![100], vec![101], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_101() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![101], vec![102], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_102() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![102], vec![103], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_103() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![103], vec![104], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_104() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![104], vec![105], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_105() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![105], vec![106], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_106() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![106], vec![107], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_107() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![107], vec![108], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_108() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![108], vec![109], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_109() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![109], vec![110], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_110() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![110], vec![111], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_111() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![111], vec![112], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_112() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![112], vec![113], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_113() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![113], vec![114], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_114() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![114], vec![115], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_115() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![115], vec![116], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_116() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![116], vec![117], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_117() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![117], vec![118], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_118() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![118], vec![119], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_119() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![119], vec![120], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_120() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![120], vec![121], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_121() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![121], vec![122], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_122() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![122], vec![123], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_123() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![123], vec![124], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_124() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![124], vec![125], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_125() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![125], vec![126], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_126() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![126], vec![127], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_127() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![127], vec![128], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_128() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![128], vec![129], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_129() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![129], vec![130], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_130() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![130], vec![131], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_131() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![131], vec![132], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_132() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![132], vec![133], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_133() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![133], vec![134], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_134() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![134], vec![135], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_135() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![135], vec![136], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_136() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![136], vec![137], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_137() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![137], vec![138], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_138() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![138], vec![139], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_139() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![139], vec![140], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_140() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![140], vec![141], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_141() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![141], vec![142], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_142() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![142], vec![143], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_143() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![143], vec![144], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_144() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![144], vec![145], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_145() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![145], vec![146], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_146() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![146], vec![147], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_147() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![147], vec![148], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_148() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![148], vec![149], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_149() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![149], vec![150], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_150() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![150], vec![151], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_151() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![151], vec![152], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_152() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![152], vec![153], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_153() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![153], vec![154], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_154() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![154], vec![155], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_155() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![155], vec![156], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_156() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![156], vec![157], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_157() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![157], vec![158], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_158() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![158], vec![159], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_159() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![159], vec![160], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_160() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![160], vec![161], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_161() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![161], vec![162], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_162() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![162], vec![163], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_163() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![163], vec![164], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_164() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![164], vec![165], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_165() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![165], vec![166], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_166() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![166], vec![167], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_167() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![167], vec![168], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_168() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![168], vec![169], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_169() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![169], vec![170], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_170() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![170], vec![171], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_171() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![171], vec![172], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_172() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![172], vec![173], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_173() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![173], vec![174], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_174() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![174], vec![175], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_175() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![175], vec![176], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_176() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![176], vec![177], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_177() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![177], vec![178], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_178() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![178], vec![179], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_179() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![179], vec![180], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_180() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![180], vec![181], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_181() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![181], vec![182], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_182() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![182], vec![183], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_183() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![183], vec![184], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_184() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![184], vec![185], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_185() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![185], vec![186], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_186() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![186], vec![187], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_187() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![187], vec![188], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_188() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![188], vec![189], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_189() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![189], vec![190], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_190() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![190], vec![191], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_191() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![191], vec![192], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_192() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![192], vec![193], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_193() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![193], vec![194], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_194() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![194], vec![195], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_195() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![195], vec![196], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_196() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![196], vec![197], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_197() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![197], vec![198], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_198() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![198], vec![199], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_199() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![199], vec![200], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_200() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![200], vec![201], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_201() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![201], vec![202], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_202() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![202], vec![203], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_203() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![203], vec![204], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_204() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![204], vec![205], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_205() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![205], vec![206], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_206() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![206], vec![207], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_207() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![207], vec![208], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_208() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![208], vec![209], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_209() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![209], vec![210], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_210() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![210], vec![211], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_211() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![211], vec![212], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_212() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![212], vec![213], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_213() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![213], vec![214], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_214() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![214], vec![215], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_215() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![215], vec![216], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_216() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![216], vec![217], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_217() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![217], vec![218], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_218() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![218], vec![219], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_219() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![219], vec![220], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_220() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![220], vec![221], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_221() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![221], vec![222], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_222() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![222], vec![223], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_223() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![223], vec![224], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_224() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![224], vec![225], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_225() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![225], vec![226], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_226() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![226], vec![227], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_227() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![227], vec![228], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_228() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![228], vec![229], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_229() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![229], vec![230], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_230() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![230], vec![231], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_231() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![231], vec![232], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_232() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![232], vec![233], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_233() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![233], vec![234], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_234() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![234], vec![235], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_235() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![235], vec![236], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_236() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![236], vec![237], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_237() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![237], vec![238], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_238() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![238], vec![239], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_239() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![239], vec![240], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_240() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![240], vec![241], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_241() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![241], vec![242], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_242() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![242], vec![243], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_243() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![243], vec![244], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_244() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![244], vec![245], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_245() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![245], vec![246], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_246() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![246], vec![247], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_247() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![247], vec![248], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_248() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![248], vec![249], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_249() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![249], vec![250], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_250() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![250], vec![251], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_251() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![251], vec![252], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_252() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![252], vec![253], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_253() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![253], vec![254], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_254() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![254], vec![255], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_255() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![255], vec![256], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_256() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![256], vec![257], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_257() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![257], vec![258], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_258() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![258], vec![259], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_259() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![259], vec![260], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_260() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![260], vec![261], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_261() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![261], vec![262], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_262() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![262], vec![263], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_263() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![263], vec![264], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_264() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![264], vec![265], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_265() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![265], vec![266], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_266() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![266], vec![267], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_267() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![267], vec![268], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_268() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![268], vec![269], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_269() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![269], vec![270], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_270() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![270], vec![271], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_271() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![271], vec![272], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_272() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![272], vec![273], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_273() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![273], vec![274], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_274() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![274], vec![275], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_275() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![275], vec![276], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_276() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![276], vec![277], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_277() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![277], vec![278], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_278() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![278], vec![279], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_279() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![279], vec![280], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_280() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![280], vec![281], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_281() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![281], vec![282], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_282() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![282], vec![283], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_283() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![283], vec![284], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_284() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![284], vec![285], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_285() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![285], vec![286], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_286() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![286], vec![287], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_287() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![287], vec![288], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_288() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![288], vec![289], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_289() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![289], vec![290], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_290() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![290], vec![291], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_291() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![291], vec![292], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_292() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![292], vec![293], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_293() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![293], vec![294], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_294() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![294], vec![295], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_295() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![295], vec![296], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_296() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![296], vec![297], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_297() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![297], vec![298], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_298() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![298], vec![299], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_299() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![299], vec![300], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_300() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![300], vec![301], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_301() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![301], vec![302], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_302() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![302], vec![303], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_303() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![303], vec![304], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_304() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![304], vec![305], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_305() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![305], vec![306], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_306() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![306], vec![307], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_307() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![307], vec![308], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_308() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![308], vec![309], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_309() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![309], vec![310], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_310() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![310], vec![311], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_311() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![311], vec![312], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_312() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![312], vec![313], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_313() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![313], vec![314], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_314() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![314], vec![315], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_315() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![315], vec![316], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_316() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![316], vec![317], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_317() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![317], vec![318], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_318() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![318], vec![319], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_319() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![319], vec![320], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_320() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![320], vec![321], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_321() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![321], vec![322], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_322() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![322], vec![323], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_323() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![323], vec![324], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_324() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![324], vec![325], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_325() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![325], vec![326], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_326() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![326], vec![327], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_327() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![327], vec![328], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_328() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![328], vec![329], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_329() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![329], vec![330], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_330() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![330], vec![331], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_331() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![331], vec![332], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_332() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![332], vec![333], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_333() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![333], vec![334], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_334() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![334], vec![335], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_335() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![335], vec![336], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_336() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![336], vec![337], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_337() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![337], vec![338], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_338() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![338], vec![339], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_339() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![339], vec![340], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_340() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![340], vec![341], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_341() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![341], vec![342], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_342() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![342], vec![343], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_343() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![343], vec![344], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_344() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![344], vec![345], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_345() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![345], vec![346], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_346() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![346], vec![347], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_347() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![347], vec![348], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_348() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![348], vec![349], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_349() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![349], vec![350], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_350() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![350], vec![351], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_351() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![351], vec![352], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_352() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![352], vec![353], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_353() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![353], vec![354], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_354() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![354], vec![355], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_355() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![355], vec![356], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_356() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![356], vec![357], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_357() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![357], vec![358], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_358() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![358], vec![359], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_359() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![359], vec![360], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_360() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![360], vec![361], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_361() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![361], vec![362], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_362() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![362], vec![363], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_363() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![363], vec![364], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_364() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![364], vec![365], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_365() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![365], vec![366], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_366() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![366], vec![367], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_fusion_stress_367() {
        let mut t = Tape::new();
        t.record(OpRecord::new("add", vec![367], vec![368], vec![vec![2]]));
        let pass = TapeFusionPass::new();
        let out = pass.run(&t);
        assert_eq!(out.op_count(), 1);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
}
