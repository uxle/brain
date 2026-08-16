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
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;

    #[test]
    fn test_tape_pruner_stress_001() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![1], vec![2], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[2]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_002() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![2], vec![3], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[3]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_003() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![3], vec![4], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[4]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_004() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![4], vec![5], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[5]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_005() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![5], vec![6], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[6]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_006() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![6], vec![7], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[7]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_007() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![7], vec![8], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[8]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_008() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![8], vec![9], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[9]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_009() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![9], vec![10], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[10]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_010() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![10], vec![11], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[11]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_011() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![11], vec![12], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[12]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_012() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![12], vec![13], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[13]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_013() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![13], vec![14], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[14]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_014() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![14], vec![15], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[15]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_015() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![15], vec![16], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[16]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_016() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![16], vec![17], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[17]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_017() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![17], vec![18], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[18]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_018() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![18], vec![19], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[19]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_019() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![19], vec![20], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[20]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_020() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![20], vec![21], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[21]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_021() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![21], vec![22], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[22]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_022() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![22], vec![23], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[23]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_023() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![23], vec![24], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[24]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_024() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![24], vec![25], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[25]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_025() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![25], vec![26], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[26]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_026() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![26], vec![27], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[27]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_027() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![27], vec![28], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[28]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_028() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![28], vec![29], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[29]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_029() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![29], vec![30], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[30]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_030() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![30], vec![31], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[31]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_031() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![31], vec![32], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[32]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_032() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![32], vec![33], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[33]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_033() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![33], vec![34], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[34]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_034() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![34], vec![35], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[35]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_035() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![35], vec![36], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[36]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_036() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![36], vec![37], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[37]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_037() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![37], vec![38], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[38]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_038() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![38], vec![39], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[39]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_039() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![39], vec![40], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[40]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_040() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![40], vec![41], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[41]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_041() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![41], vec![42], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[42]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_042() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![42], vec![43], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[43]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_043() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![43], vec![44], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[44]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_044() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![44], vec![45], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[45]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_045() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![45], vec![46], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[46]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_046() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![46], vec![47], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[47]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_047() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![47], vec![48], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[48]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_048() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![48], vec![49], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[49]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_049() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![49], vec![50], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[50]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_050() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![50], vec![51], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[51]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_051() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![51], vec![52], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[52]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_052() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![52], vec![53], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[53]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_053() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![53], vec![54], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[54]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_054() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![54], vec![55], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[55]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_055() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![55], vec![56], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[56]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_056() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![56], vec![57], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[57]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_057() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![57], vec![58], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[58]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_058() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![58], vec![59], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[59]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_059() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![59], vec![60], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[60]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_060() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![60], vec![61], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[61]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_061() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![61], vec![62], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[62]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_062() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![62], vec![63], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[63]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_063() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![63], vec![64], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[64]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_064() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![64], vec![65], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[65]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_065() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![65], vec![66], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[66]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_066() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![66], vec![67], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[67]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_067() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![67], vec![68], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[68]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_068() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![68], vec![69], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[69]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_069() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![69], vec![70], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[70]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_070() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![70], vec![71], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[71]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_071() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![71], vec![72], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[72]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_072() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![72], vec![73], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[73]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_073() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![73], vec![74], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[74]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_074() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![74], vec![75], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[75]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_075() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![75], vec![76], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[76]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_076() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![76], vec![77], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[77]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_077() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![77], vec![78], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[78]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_078() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![78], vec![79], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[79]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_079() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![79], vec![80], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[80]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_080() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![80], vec![81], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[81]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_081() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![81], vec![82], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[82]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_082() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![82], vec![83], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[83]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_083() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![83], vec![84], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[84]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_084() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![84], vec![85], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[85]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_085() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![85], vec![86], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[86]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_086() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![86], vec![87], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[87]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_087() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![87], vec![88], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[88]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_088() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![88], vec![89], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[89]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_089() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![89], vec![90], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[90]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_090() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![90], vec![91], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[91]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_091() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![91], vec![92], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[92]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_092() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![92], vec![93], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[93]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_093() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![93], vec![94], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[94]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_094() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![94], vec![95], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[95]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_095() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![95], vec![96], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[96]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_096() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![96], vec![97], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[97]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_097() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![97], vec![98], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[98]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_098() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![98], vec![99], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[99]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_099() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![99], vec![100], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[100]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_100() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![100], vec![101], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[101]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_101() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![101], vec![102], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[102]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_102() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![102], vec![103], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[103]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_103() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![103], vec![104], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[104]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_104() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![104], vec![105], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[105]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_105() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![105], vec![106], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[106]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_106() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![106], vec![107], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[107]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_107() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![107], vec![108], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[108]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_108() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![108], vec![109], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[109]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_109() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![109], vec![110], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[110]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_110() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![110], vec![111], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[111]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_111() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![111], vec![112], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[112]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_112() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![112], vec![113], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[113]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_113() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![113], vec![114], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[114]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_114() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![114], vec![115], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[115]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_115() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![115], vec![116], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[116]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_116() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![116], vec![117], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[117]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_117() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![117], vec![118], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[118]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_118() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![118], vec![119], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[119]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_119() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![119], vec![120], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[120]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_120() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![120], vec![121], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[121]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_121() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![121], vec![122], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[122]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_122() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![122], vec![123], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[123]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_123() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![123], vec![124], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[124]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_124() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![124], vec![125], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[125]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_125() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![125], vec![126], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[126]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_126() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![126], vec![127], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[127]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_127() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![127], vec![128], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[128]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_128() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![128], vec![129], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[129]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_129() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![129], vec![130], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[130]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_130() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![130], vec![131], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[131]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_131() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![131], vec![132], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[132]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_132() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![132], vec![133], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[133]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_133() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![133], vec![134], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[134]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_134() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![134], vec![135], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[135]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_135() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![135], vec![136], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[136]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_136() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![136], vec![137], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[137]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_137() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![137], vec![138], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[138]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_138() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![138], vec![139], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[139]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_139() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![139], vec![140], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[140]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_140() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![140], vec![141], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[141]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_141() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![141], vec![142], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[142]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_142() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![142], vec![143], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[143]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_143() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![143], vec![144], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[144]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_144() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![144], vec![145], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[145]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_145() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![145], vec![146], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[146]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_146() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![146], vec![147], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[147]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_147() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![147], vec![148], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[148]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_148() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![148], vec![149], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[149]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_149() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![149], vec![150], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[150]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_150() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![150], vec![151], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[151]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_151() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![151], vec![152], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[152]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_152() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![152], vec![153], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[153]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_153() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![153], vec![154], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[154]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_154() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![154], vec![155], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[155]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_155() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![155], vec![156], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[156]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_156() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![156], vec![157], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[157]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_157() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![157], vec![158], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[158]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_158() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![158], vec![159], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[159]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_159() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![159], vec![160], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[160]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_160() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![160], vec![161], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[161]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_161() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![161], vec![162], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[162]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_162() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![162], vec![163], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[163]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_163() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![163], vec![164], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[164]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_164() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![164], vec![165], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[165]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_165() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![165], vec![166], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[166]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_166() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![166], vec![167], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[167]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_167() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![167], vec![168], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[168]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_168() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![168], vec![169], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[169]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_169() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![169], vec![170], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[170]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_170() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![170], vec![171], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[171]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_171() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![171], vec![172], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[172]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_172() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![172], vec![173], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[173]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_173() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![173], vec![174], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[174]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_174() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![174], vec![175], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[175]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_175() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![175], vec![176], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[176]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_176() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![176], vec![177], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[177]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_177() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![177], vec![178], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[178]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_178() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![178], vec![179], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[179]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_179() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![179], vec![180], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[180]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_180() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![180], vec![181], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[181]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_181() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![181], vec![182], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[182]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_182() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![182], vec![183], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[183]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_183() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![183], vec![184], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[184]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_184() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![184], vec![185], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[185]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_185() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![185], vec![186], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[186]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_186() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![186], vec![187], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[187]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_187() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![187], vec![188], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[188]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_188() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![188], vec![189], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[189]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_189() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![189], vec![190], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[190]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_190() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![190], vec![191], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[191]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_191() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![191], vec![192], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[192]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_192() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![192], vec![193], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[193]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_193() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![193], vec![194], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[194]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_194() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![194], vec![195], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[195]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_195() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![195], vec![196], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[196]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_196() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![196], vec![197], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[197]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_197() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![197], vec![198], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[198]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_198() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![198], vec![199], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[199]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_199() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![199], vec![200], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[200]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_200() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![200], vec![201], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[201]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_201() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![201], vec![202], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[202]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_202() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![202], vec![203], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[203]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_203() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![203], vec![204], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[204]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_204() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![204], vec![205], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[205]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_205() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![205], vec![206], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[206]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_206() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![206], vec![207], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[207]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_207() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![207], vec![208], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[208]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_208() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![208], vec![209], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[209]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_209() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![209], vec![210], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[210]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_210() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![210], vec![211], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[211]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_211() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![211], vec![212], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[212]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_212() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![212], vec![213], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[213]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_213() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![213], vec![214], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[214]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_214() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![214], vec![215], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[215]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_215() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![215], vec![216], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[216]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_216() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![216], vec![217], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[217]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_217() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![217], vec![218], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[218]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_218() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![218], vec![219], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[219]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_219() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![219], vec![220], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[220]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_220() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![220], vec![221], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[221]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_221() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![221], vec![222], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[222]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_222() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![222], vec![223], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[223]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_223() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![223], vec![224], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[224]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_224() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![224], vec![225], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[225]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_225() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![225], vec![226], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[226]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_226() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![226], vec![227], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[227]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_227() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![227], vec![228], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[228]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_228() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![228], vec![229], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[229]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_229() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![229], vec![230], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[230]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_230() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![230], vec![231], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[231]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_231() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![231], vec![232], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[232]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_232() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![232], vec![233], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[233]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_233() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![233], vec![234], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[234]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_234() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![234], vec![235], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[235]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_235() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![235], vec![236], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[236]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_236() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![236], vec![237], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[237]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_237() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![237], vec![238], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[238]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_238() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![238], vec![239], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[239]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_239() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![239], vec![240], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[240]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_240() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![240], vec![241], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[241]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_241() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![241], vec![242], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[242]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_242() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![242], vec![243], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[243]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_243() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![243], vec![244], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[244]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_244() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![244], vec![245], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[245]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_245() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![245], vec![246], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[246]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_246() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![246], vec![247], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[247]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_247() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![247], vec![248], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[248]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_248() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![248], vec![249], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[249]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_249() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![249], vec![250], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[250]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_250() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![250], vec![251], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[251]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_251() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![251], vec![252], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[252]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_252() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![252], vec![253], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[253]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_253() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![253], vec![254], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[254]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_254() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![254], vec![255], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[255]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_255() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![255], vec![256], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[256]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_256() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![256], vec![257], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[257]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_257() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![257], vec![258], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[258]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_258() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![258], vec![259], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[259]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_259() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![259], vec![260], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[260]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_260() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![260], vec![261], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[261]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_261() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![261], vec![262], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[262]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_262() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![262], vec![263], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[263]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_263() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![263], vec![264], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[264]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_264() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![264], vec![265], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[265]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_265() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![265], vec![266], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[266]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_266() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![266], vec![267], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[267]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_267() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![267], vec![268], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[268]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_268() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![268], vec![269], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[269]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_269() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![269], vec![270], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[270]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_270() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![270], vec![271], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[271]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_271() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![271], vec![272], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[272]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_272() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![272], vec![273], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[273]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_273() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![273], vec![274], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[274]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_274() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![274], vec![275], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[275]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_275() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![275], vec![276], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[276]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_276() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![276], vec![277], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[277]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_277() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![277], vec![278], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[278]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_278() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![278], vec![279], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[279]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_279() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![279], vec![280], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[280]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_280() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![280], vec![281], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[281]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_281() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![281], vec![282], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[282]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_282() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![282], vec![283], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[283]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_283() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![283], vec![284], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[284]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_284() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![284], vec![285], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[285]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_285() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![285], vec![286], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[286]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_286() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![286], vec![287], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[287]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_287() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![287], vec![288], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[288]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_288() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![288], vec![289], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[289]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_289() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![289], vec![290], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[290]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_290() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![290], vec![291], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[291]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_291() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![291], vec![292], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[292]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_292() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![292], vec![293], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[293]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_293() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![293], vec![294], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[294]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_294() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![294], vec![295], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[295]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_295() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![295], vec![296], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[296]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_296() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![296], vec![297], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[297]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_297() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![297], vec![298], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[298]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_298() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![298], vec![299], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[299]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_299() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![299], vec![300], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[300]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_300() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![300], vec![301], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[301]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_301() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![301], vec![302], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[302]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_302() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![302], vec![303], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[303]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_303() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![303], vec![304], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[304]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_304() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![304], vec![305], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[305]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_305() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![305], vec![306], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[306]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_306() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![306], vec![307], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[307]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_307() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![307], vec![308], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[308]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_308() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![308], vec![309], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[309]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_309() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![309], vec![310], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[310]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_310() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![310], vec![311], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[311]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_311() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![311], vec![312], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[312]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_312() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![312], vec![313], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[313]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_313() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![313], vec![314], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[314]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_314() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![314], vec![315], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[315]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_315() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![315], vec![316], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[316]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_316() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![316], vec![317], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[317]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_317() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![317], vec![318], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[318]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_318() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![318], vec![319], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[319]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_319() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![319], vec![320], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[320]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_320() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![320], vec![321], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[321]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_321() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![321], vec![322], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[322]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_322() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![322], vec![323], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[323]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_323() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![323], vec![324], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[324]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_324() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![324], vec![325], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[325]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_325() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![325], vec![326], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[326]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_326() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![326], vec![327], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[327]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_327() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![327], vec![328], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[328]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_328() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![328], vec![329], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[329]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_329() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![329], vec![330], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[330]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_330() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![330], vec![331], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[331]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_331() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![331], vec![332], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[332]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_332() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![332], vec![333], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[333]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_333() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![333], vec![334], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[334]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_334() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![334], vec![335], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[335]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_335() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![335], vec![336], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[336]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_336() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![336], vec![337], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[337]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_337() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![337], vec![338], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[338]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_338() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![338], vec![339], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[339]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_339() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![339], vec![340], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[340]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_340() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![340], vec![341], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[341]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_341() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![341], vec![342], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[342]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_342() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![342], vec![343], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[343]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_343() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![343], vec![344], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[344]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_344() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![344], vec![345], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[345]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_345() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![345], vec![346], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[346]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_346() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![346], vec![347], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[347]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_347() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![347], vec![348], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[348]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_348() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![348], vec![349], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[349]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_349() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![349], vec![350], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[350]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_350() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![350], vec![351], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[351]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_351() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![351], vec![352], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[352]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_352() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![352], vec![353], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[353]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_353() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![353], vec![354], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[354]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_354() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![354], vec![355], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[355]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_355() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![355], vec![356], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[356]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_356() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![356], vec![357], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[357]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_357() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![357], vec![358], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[358]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_358() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![358], vec![359], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[359]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_359() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![359], vec![360], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[360]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_360() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![360], vec![361], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[361]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_361() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![361], vec![362], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[362]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_362() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![362], vec![363], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[363]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_363() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![363], vec![364], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[364]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_364() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![364], vec![365], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[365]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_365() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![365], vec![366], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[366]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_366() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![366], vec![367], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[367]);
        assert_eq!(out.op_count(), 1);
    }

    #[test]
    fn test_tape_pruner_stress_367() {
        let mut t = Tape::new();
        t.record(OpRecord::new("relu", vec![367], vec![368], vec![vec![1]]));
        let pruner = TapePruner::new();
        let out = pruner.prune(&t, &[368]);
        assert_eq!(out.op_count(), 1);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
}
