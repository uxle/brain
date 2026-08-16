//! # Tape Record Nodes
//!
//! Strongly-typed records representing operation executions on the autograd tape.

/// An operation record stored on the execution tape.
#[derive(Debug, Clone, PartialEq)]
pub struct OpRecord {
    /// Name of the executed operation.
    pub op_name: String,
    /// IDs of input nodes.
    pub inputs: Vec<usize>,
    /// IDs of output nodes.
    pub outputs: Vec<usize>,
    /// Output tensor shapes.
    pub shapes: Vec<Vec<usize>>,
}

impl OpRecord {
    /// Creates a new op record.
    pub fn new(op_name: impl Into<String>, inputs: Vec<usize>, outputs: Vec<usize>, shapes: Vec<Vec<usize>>) -> Self {
        Self {
            op_name: op_name.into(),
            inputs,
            outputs,
            shapes,
        }
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
    fn test_tape_op_record_stress_001() {
        let rec = OpRecord::new("add", vec![1, 2], vec![3], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_002() {
        let rec = OpRecord::new("add", vec![2, 3], vec![4], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_003() {
        let rec = OpRecord::new("add", vec![3, 4], vec![5], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_004() {
        let rec = OpRecord::new("add", vec![4, 5], vec![6], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_005() {
        let rec = OpRecord::new("add", vec![5, 6], vec![7], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_006() {
        let rec = OpRecord::new("add", vec![6, 7], vec![8], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_007() {
        let rec = OpRecord::new("add", vec![7, 8], vec![9], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_008() {
        let rec = OpRecord::new("add", vec![8, 9], vec![10], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_009() {
        let rec = OpRecord::new("add", vec![9, 10], vec![11], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_010() {
        let rec = OpRecord::new("add", vec![10, 11], vec![12], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_011() {
        let rec = OpRecord::new("add", vec![11, 12], vec![13], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_012() {
        let rec = OpRecord::new("add", vec![12, 13], vec![14], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_013() {
        let rec = OpRecord::new("add", vec![13, 14], vec![15], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_014() {
        let rec = OpRecord::new("add", vec![14, 15], vec![16], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_015() {
        let rec = OpRecord::new("add", vec![15, 16], vec![17], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_016() {
        let rec = OpRecord::new("add", vec![16, 17], vec![18], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_017() {
        let rec = OpRecord::new("add", vec![17, 18], vec![19], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_018() {
        let rec = OpRecord::new("add", vec![18, 19], vec![20], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_019() {
        let rec = OpRecord::new("add", vec![19, 20], vec![21], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_020() {
        let rec = OpRecord::new("add", vec![20, 21], vec![22], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_021() {
        let rec = OpRecord::new("add", vec![21, 22], vec![23], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_022() {
        let rec = OpRecord::new("add", vec![22, 23], vec![24], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_023() {
        let rec = OpRecord::new("add", vec![23, 24], vec![25], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_024() {
        let rec = OpRecord::new("add", vec![24, 25], vec![26], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_025() {
        let rec = OpRecord::new("add", vec![25, 26], vec![27], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_026() {
        let rec = OpRecord::new("add", vec![26, 27], vec![28], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_027() {
        let rec = OpRecord::new("add", vec![27, 28], vec![29], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_028() {
        let rec = OpRecord::new("add", vec![28, 29], vec![30], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_029() {
        let rec = OpRecord::new("add", vec![29, 30], vec![31], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_030() {
        let rec = OpRecord::new("add", vec![30, 31], vec![32], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_031() {
        let rec = OpRecord::new("add", vec![31, 32], vec![33], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_032() {
        let rec = OpRecord::new("add", vec![32, 33], vec![34], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_033() {
        let rec = OpRecord::new("add", vec![33, 34], vec![35], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_034() {
        let rec = OpRecord::new("add", vec![34, 35], vec![36], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_035() {
        let rec = OpRecord::new("add", vec![35, 36], vec![37], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_036() {
        let rec = OpRecord::new("add", vec![36, 37], vec![38], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_037() {
        let rec = OpRecord::new("add", vec![37, 38], vec![39], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_038() {
        let rec = OpRecord::new("add", vec![38, 39], vec![40], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_039() {
        let rec = OpRecord::new("add", vec![39, 40], vec![41], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_040() {
        let rec = OpRecord::new("add", vec![40, 41], vec![42], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_041() {
        let rec = OpRecord::new("add", vec![41, 42], vec![43], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_042() {
        let rec = OpRecord::new("add", vec![42, 43], vec![44], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_043() {
        let rec = OpRecord::new("add", vec![43, 44], vec![45], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_044() {
        let rec = OpRecord::new("add", vec![44, 45], vec![46], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_045() {
        let rec = OpRecord::new("add", vec![45, 46], vec![47], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_046() {
        let rec = OpRecord::new("add", vec![46, 47], vec![48], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_047() {
        let rec = OpRecord::new("add", vec![47, 48], vec![49], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_048() {
        let rec = OpRecord::new("add", vec![48, 49], vec![50], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_049() {
        let rec = OpRecord::new("add", vec![49, 50], vec![51], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_050() {
        let rec = OpRecord::new("add", vec![50, 51], vec![52], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_051() {
        let rec = OpRecord::new("add", vec![51, 52], vec![53], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_052() {
        let rec = OpRecord::new("add", vec![52, 53], vec![54], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_053() {
        let rec = OpRecord::new("add", vec![53, 54], vec![55], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_054() {
        let rec = OpRecord::new("add", vec![54, 55], vec![56], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_055() {
        let rec = OpRecord::new("add", vec![55, 56], vec![57], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_056() {
        let rec = OpRecord::new("add", vec![56, 57], vec![58], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_057() {
        let rec = OpRecord::new("add", vec![57, 58], vec![59], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_058() {
        let rec = OpRecord::new("add", vec![58, 59], vec![60], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_059() {
        let rec = OpRecord::new("add", vec![59, 60], vec![61], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_060() {
        let rec = OpRecord::new("add", vec![60, 61], vec![62], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_061() {
        let rec = OpRecord::new("add", vec![61, 62], vec![63], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_062() {
        let rec = OpRecord::new("add", vec![62, 63], vec![64], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_063() {
        let rec = OpRecord::new("add", vec![63, 64], vec![65], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_064() {
        let rec = OpRecord::new("add", vec![64, 65], vec![66], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_065() {
        let rec = OpRecord::new("add", vec![65, 66], vec![67], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_066() {
        let rec = OpRecord::new("add", vec![66, 67], vec![68], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_067() {
        let rec = OpRecord::new("add", vec![67, 68], vec![69], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_068() {
        let rec = OpRecord::new("add", vec![68, 69], vec![70], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_069() {
        let rec = OpRecord::new("add", vec![69, 70], vec![71], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_070() {
        let rec = OpRecord::new("add", vec![70, 71], vec![72], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_071() {
        let rec = OpRecord::new("add", vec![71, 72], vec![73], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_072() {
        let rec = OpRecord::new("add", vec![72, 73], vec![74], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_073() {
        let rec = OpRecord::new("add", vec![73, 74], vec![75], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_074() {
        let rec = OpRecord::new("add", vec![74, 75], vec![76], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_075() {
        let rec = OpRecord::new("add", vec![75, 76], vec![77], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_076() {
        let rec = OpRecord::new("add", vec![76, 77], vec![78], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_077() {
        let rec = OpRecord::new("add", vec![77, 78], vec![79], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_078() {
        let rec = OpRecord::new("add", vec![78, 79], vec![80], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_079() {
        let rec = OpRecord::new("add", vec![79, 80], vec![81], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_080() {
        let rec = OpRecord::new("add", vec![80, 81], vec![82], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_081() {
        let rec = OpRecord::new("add", vec![81, 82], vec![83], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_082() {
        let rec = OpRecord::new("add", vec![82, 83], vec![84], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_083() {
        let rec = OpRecord::new("add", vec![83, 84], vec![85], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_084() {
        let rec = OpRecord::new("add", vec![84, 85], vec![86], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_085() {
        let rec = OpRecord::new("add", vec![85, 86], vec![87], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_086() {
        let rec = OpRecord::new("add", vec![86, 87], vec![88], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_087() {
        let rec = OpRecord::new("add", vec![87, 88], vec![89], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_088() {
        let rec = OpRecord::new("add", vec![88, 89], vec![90], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_089() {
        let rec = OpRecord::new("add", vec![89, 90], vec![91], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_090() {
        let rec = OpRecord::new("add", vec![90, 91], vec![92], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_091() {
        let rec = OpRecord::new("add", vec![91, 92], vec![93], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_092() {
        let rec = OpRecord::new("add", vec![92, 93], vec![94], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_093() {
        let rec = OpRecord::new("add", vec![93, 94], vec![95], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_094() {
        let rec = OpRecord::new("add", vec![94, 95], vec![96], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_095() {
        let rec = OpRecord::new("add", vec![95, 96], vec![97], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_096() {
        let rec = OpRecord::new("add", vec![96, 97], vec![98], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_097() {
        let rec = OpRecord::new("add", vec![97, 98], vec![99], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_098() {
        let rec = OpRecord::new("add", vec![98, 99], vec![100], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_099() {
        let rec = OpRecord::new("add", vec![99, 100], vec![101], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_100() {
        let rec = OpRecord::new("add", vec![100, 101], vec![102], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_101() {
        let rec = OpRecord::new("add", vec![101, 102], vec![103], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_102() {
        let rec = OpRecord::new("add", vec![102, 103], vec![104], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_103() {
        let rec = OpRecord::new("add", vec![103, 104], vec![105], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_104() {
        let rec = OpRecord::new("add", vec![104, 105], vec![106], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_105() {
        let rec = OpRecord::new("add", vec![105, 106], vec![107], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_106() {
        let rec = OpRecord::new("add", vec![106, 107], vec![108], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_107() {
        let rec = OpRecord::new("add", vec![107, 108], vec![109], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_108() {
        let rec = OpRecord::new("add", vec![108, 109], vec![110], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_109() {
        let rec = OpRecord::new("add", vec![109, 110], vec![111], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_110() {
        let rec = OpRecord::new("add", vec![110, 111], vec![112], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_111() {
        let rec = OpRecord::new("add", vec![111, 112], vec![113], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_112() {
        let rec = OpRecord::new("add", vec![112, 113], vec![114], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_113() {
        let rec = OpRecord::new("add", vec![113, 114], vec![115], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_114() {
        let rec = OpRecord::new("add", vec![114, 115], vec![116], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_115() {
        let rec = OpRecord::new("add", vec![115, 116], vec![117], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_116() {
        let rec = OpRecord::new("add", vec![116, 117], vec![118], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_117() {
        let rec = OpRecord::new("add", vec![117, 118], vec![119], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_118() {
        let rec = OpRecord::new("add", vec![118, 119], vec![120], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_119() {
        let rec = OpRecord::new("add", vec![119, 120], vec![121], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_120() {
        let rec = OpRecord::new("add", vec![120, 121], vec![122], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_121() {
        let rec = OpRecord::new("add", vec![121, 122], vec![123], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_122() {
        let rec = OpRecord::new("add", vec![122, 123], vec![124], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_123() {
        let rec = OpRecord::new("add", vec![123, 124], vec![125], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_124() {
        let rec = OpRecord::new("add", vec![124, 125], vec![126], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_125() {
        let rec = OpRecord::new("add", vec![125, 126], vec![127], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_126() {
        let rec = OpRecord::new("add", vec![126, 127], vec![128], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_127() {
        let rec = OpRecord::new("add", vec![127, 128], vec![129], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_128() {
        let rec = OpRecord::new("add", vec![128, 129], vec![130], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_129() {
        let rec = OpRecord::new("add", vec![129, 130], vec![131], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_130() {
        let rec = OpRecord::new("add", vec![130, 131], vec![132], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_131() {
        let rec = OpRecord::new("add", vec![131, 132], vec![133], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_132() {
        let rec = OpRecord::new("add", vec![132, 133], vec![134], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_133() {
        let rec = OpRecord::new("add", vec![133, 134], vec![135], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_134() {
        let rec = OpRecord::new("add", vec![134, 135], vec![136], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_135() {
        let rec = OpRecord::new("add", vec![135, 136], vec![137], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_136() {
        let rec = OpRecord::new("add", vec![136, 137], vec![138], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_137() {
        let rec = OpRecord::new("add", vec![137, 138], vec![139], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_138() {
        let rec = OpRecord::new("add", vec![138, 139], vec![140], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_139() {
        let rec = OpRecord::new("add", vec![139, 140], vec![141], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_140() {
        let rec = OpRecord::new("add", vec![140, 141], vec![142], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_141() {
        let rec = OpRecord::new("add", vec![141, 142], vec![143], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_142() {
        let rec = OpRecord::new("add", vec![142, 143], vec![144], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_143() {
        let rec = OpRecord::new("add", vec![143, 144], vec![145], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_144() {
        let rec = OpRecord::new("add", vec![144, 145], vec![146], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_145() {
        let rec = OpRecord::new("add", vec![145, 146], vec![147], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_146() {
        let rec = OpRecord::new("add", vec![146, 147], vec![148], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_147() {
        let rec = OpRecord::new("add", vec![147, 148], vec![149], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_148() {
        let rec = OpRecord::new("add", vec![148, 149], vec![150], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_149() {
        let rec = OpRecord::new("add", vec![149, 150], vec![151], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_150() {
        let rec = OpRecord::new("add", vec![150, 151], vec![152], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_151() {
        let rec = OpRecord::new("add", vec![151, 152], vec![153], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_152() {
        let rec = OpRecord::new("add", vec![152, 153], vec![154], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_153() {
        let rec = OpRecord::new("add", vec![153, 154], vec![155], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_154() {
        let rec = OpRecord::new("add", vec![154, 155], vec![156], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_155() {
        let rec = OpRecord::new("add", vec![155, 156], vec![157], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_156() {
        let rec = OpRecord::new("add", vec![156, 157], vec![158], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_157() {
        let rec = OpRecord::new("add", vec![157, 158], vec![159], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_158() {
        let rec = OpRecord::new("add", vec![158, 159], vec![160], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_159() {
        let rec = OpRecord::new("add", vec![159, 160], vec![161], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_160() {
        let rec = OpRecord::new("add", vec![160, 161], vec![162], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_161() {
        let rec = OpRecord::new("add", vec![161, 162], vec![163], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_162() {
        let rec = OpRecord::new("add", vec![162, 163], vec![164], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_163() {
        let rec = OpRecord::new("add", vec![163, 164], vec![165], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_164() {
        let rec = OpRecord::new("add", vec![164, 165], vec![166], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_165() {
        let rec = OpRecord::new("add", vec![165, 166], vec![167], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_166() {
        let rec = OpRecord::new("add", vec![166, 167], vec![168], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_167() {
        let rec = OpRecord::new("add", vec![167, 168], vec![169], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_168() {
        let rec = OpRecord::new("add", vec![168, 169], vec![170], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_169() {
        let rec = OpRecord::new("add", vec![169, 170], vec![171], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_170() {
        let rec = OpRecord::new("add", vec![170, 171], vec![172], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_171() {
        let rec = OpRecord::new("add", vec![171, 172], vec![173], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_172() {
        let rec = OpRecord::new("add", vec![172, 173], vec![174], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_173() {
        let rec = OpRecord::new("add", vec![173, 174], vec![175], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_174() {
        let rec = OpRecord::new("add", vec![174, 175], vec![176], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_175() {
        let rec = OpRecord::new("add", vec![175, 176], vec![177], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_176() {
        let rec = OpRecord::new("add", vec![176, 177], vec![178], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_177() {
        let rec = OpRecord::new("add", vec![177, 178], vec![179], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_178() {
        let rec = OpRecord::new("add", vec![178, 179], vec![180], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_179() {
        let rec = OpRecord::new("add", vec![179, 180], vec![181], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_180() {
        let rec = OpRecord::new("add", vec![180, 181], vec![182], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_181() {
        let rec = OpRecord::new("add", vec![181, 182], vec![183], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_182() {
        let rec = OpRecord::new("add", vec![182, 183], vec![184], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_183() {
        let rec = OpRecord::new("add", vec![183, 184], vec![185], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_184() {
        let rec = OpRecord::new("add", vec![184, 185], vec![186], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_185() {
        let rec = OpRecord::new("add", vec![185, 186], vec![187], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_186() {
        let rec = OpRecord::new("add", vec![186, 187], vec![188], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_187() {
        let rec = OpRecord::new("add", vec![187, 188], vec![189], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_188() {
        let rec = OpRecord::new("add", vec![188, 189], vec![190], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_189() {
        let rec = OpRecord::new("add", vec![189, 190], vec![191], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_190() {
        let rec = OpRecord::new("add", vec![190, 191], vec![192], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_191() {
        let rec = OpRecord::new("add", vec![191, 192], vec![193], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_192() {
        let rec = OpRecord::new("add", vec![192, 193], vec![194], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_193() {
        let rec = OpRecord::new("add", vec![193, 194], vec![195], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_194() {
        let rec = OpRecord::new("add", vec![194, 195], vec![196], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_195() {
        let rec = OpRecord::new("add", vec![195, 196], vec![197], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_196() {
        let rec = OpRecord::new("add", vec![196, 197], vec![198], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_197() {
        let rec = OpRecord::new("add", vec![197, 198], vec![199], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_198() {
        let rec = OpRecord::new("add", vec![198, 199], vec![200], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_199() {
        let rec = OpRecord::new("add", vec![199, 200], vec![201], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_200() {
        let rec = OpRecord::new("add", vec![200, 201], vec![202], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_201() {
        let rec = OpRecord::new("add", vec![201, 202], vec![203], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_202() {
        let rec = OpRecord::new("add", vec![202, 203], vec![204], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_203() {
        let rec = OpRecord::new("add", vec![203, 204], vec![205], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_204() {
        let rec = OpRecord::new("add", vec![204, 205], vec![206], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_205() {
        let rec = OpRecord::new("add", vec![205, 206], vec![207], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_206() {
        let rec = OpRecord::new("add", vec![206, 207], vec![208], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_207() {
        let rec = OpRecord::new("add", vec![207, 208], vec![209], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_208() {
        let rec = OpRecord::new("add", vec![208, 209], vec![210], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_209() {
        let rec = OpRecord::new("add", vec![209, 210], vec![211], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_210() {
        let rec = OpRecord::new("add", vec![210, 211], vec![212], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_211() {
        let rec = OpRecord::new("add", vec![211, 212], vec![213], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_212() {
        let rec = OpRecord::new("add", vec![212, 213], vec![214], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_213() {
        let rec = OpRecord::new("add", vec![213, 214], vec![215], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_214() {
        let rec = OpRecord::new("add", vec![214, 215], vec![216], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_215() {
        let rec = OpRecord::new("add", vec![215, 216], vec![217], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_216() {
        let rec = OpRecord::new("add", vec![216, 217], vec![218], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_217() {
        let rec = OpRecord::new("add", vec![217, 218], vec![219], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_218() {
        let rec = OpRecord::new("add", vec![218, 219], vec![220], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_219() {
        let rec = OpRecord::new("add", vec![219, 220], vec![221], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_220() {
        let rec = OpRecord::new("add", vec![220, 221], vec![222], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_221() {
        let rec = OpRecord::new("add", vec![221, 222], vec![223], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_222() {
        let rec = OpRecord::new("add", vec![222, 223], vec![224], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_223() {
        let rec = OpRecord::new("add", vec![223, 224], vec![225], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_224() {
        let rec = OpRecord::new("add", vec![224, 225], vec![226], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_225() {
        let rec = OpRecord::new("add", vec![225, 226], vec![227], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_226() {
        let rec = OpRecord::new("add", vec![226, 227], vec![228], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_227() {
        let rec = OpRecord::new("add", vec![227, 228], vec![229], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_228() {
        let rec = OpRecord::new("add", vec![228, 229], vec![230], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_229() {
        let rec = OpRecord::new("add", vec![229, 230], vec![231], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_230() {
        let rec = OpRecord::new("add", vec![230, 231], vec![232], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_231() {
        let rec = OpRecord::new("add", vec![231, 232], vec![233], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_232() {
        let rec = OpRecord::new("add", vec![232, 233], vec![234], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_233() {
        let rec = OpRecord::new("add", vec![233, 234], vec![235], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_234() {
        let rec = OpRecord::new("add", vec![234, 235], vec![236], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_235() {
        let rec = OpRecord::new("add", vec![235, 236], vec![237], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_236() {
        let rec = OpRecord::new("add", vec![236, 237], vec![238], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_237() {
        let rec = OpRecord::new("add", vec![237, 238], vec![239], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_238() {
        let rec = OpRecord::new("add", vec![238, 239], vec![240], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_239() {
        let rec = OpRecord::new("add", vec![239, 240], vec![241], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_240() {
        let rec = OpRecord::new("add", vec![240, 241], vec![242], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_241() {
        let rec = OpRecord::new("add", vec![241, 242], vec![243], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_242() {
        let rec = OpRecord::new("add", vec![242, 243], vec![244], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_243() {
        let rec = OpRecord::new("add", vec![243, 244], vec![245], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_244() {
        let rec = OpRecord::new("add", vec![244, 245], vec![246], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_245() {
        let rec = OpRecord::new("add", vec![245, 246], vec![247], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_246() {
        let rec = OpRecord::new("add", vec![246, 247], vec![248], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_247() {
        let rec = OpRecord::new("add", vec![247, 248], vec![249], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_248() {
        let rec = OpRecord::new("add", vec![248, 249], vec![250], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_249() {
        let rec = OpRecord::new("add", vec![249, 250], vec![251], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_250() {
        let rec = OpRecord::new("add", vec![250, 251], vec![252], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_251() {
        let rec = OpRecord::new("add", vec![251, 252], vec![253], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_252() {
        let rec = OpRecord::new("add", vec![252, 253], vec![254], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_253() {
        let rec = OpRecord::new("add", vec![253, 254], vec![255], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_254() {
        let rec = OpRecord::new("add", vec![254, 255], vec![256], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_255() {
        let rec = OpRecord::new("add", vec![255, 256], vec![257], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_256() {
        let rec = OpRecord::new("add", vec![256, 257], vec![258], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_257() {
        let rec = OpRecord::new("add", vec![257, 258], vec![259], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_258() {
        let rec = OpRecord::new("add", vec![258, 259], vec![260], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_259() {
        let rec = OpRecord::new("add", vec![259, 260], vec![261], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_260() {
        let rec = OpRecord::new("add", vec![260, 261], vec![262], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_261() {
        let rec = OpRecord::new("add", vec![261, 262], vec![263], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_262() {
        let rec = OpRecord::new("add", vec![262, 263], vec![264], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_263() {
        let rec = OpRecord::new("add", vec![263, 264], vec![265], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_264() {
        let rec = OpRecord::new("add", vec![264, 265], vec![266], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_265() {
        let rec = OpRecord::new("add", vec![265, 266], vec![267], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_266() {
        let rec = OpRecord::new("add", vec![266, 267], vec![268], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_267() {
        let rec = OpRecord::new("add", vec![267, 268], vec![269], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_268() {
        let rec = OpRecord::new("add", vec![268, 269], vec![270], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_269() {
        let rec = OpRecord::new("add", vec![269, 270], vec![271], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_270() {
        let rec = OpRecord::new("add", vec![270, 271], vec![272], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_271() {
        let rec = OpRecord::new("add", vec![271, 272], vec![273], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_272() {
        let rec = OpRecord::new("add", vec![272, 273], vec![274], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_273() {
        let rec = OpRecord::new("add", vec![273, 274], vec![275], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_274() {
        let rec = OpRecord::new("add", vec![274, 275], vec![276], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_275() {
        let rec = OpRecord::new("add", vec![275, 276], vec![277], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_276() {
        let rec = OpRecord::new("add", vec![276, 277], vec![278], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_277() {
        let rec = OpRecord::new("add", vec![277, 278], vec![279], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_278() {
        let rec = OpRecord::new("add", vec![278, 279], vec![280], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_279() {
        let rec = OpRecord::new("add", vec![279, 280], vec![281], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_280() {
        let rec = OpRecord::new("add", vec![280, 281], vec![282], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_281() {
        let rec = OpRecord::new("add", vec![281, 282], vec![283], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_282() {
        let rec = OpRecord::new("add", vec![282, 283], vec![284], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_283() {
        let rec = OpRecord::new("add", vec![283, 284], vec![285], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_284() {
        let rec = OpRecord::new("add", vec![284, 285], vec![286], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_285() {
        let rec = OpRecord::new("add", vec![285, 286], vec![287], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_286() {
        let rec = OpRecord::new("add", vec![286, 287], vec![288], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_287() {
        let rec = OpRecord::new("add", vec![287, 288], vec![289], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_288() {
        let rec = OpRecord::new("add", vec![288, 289], vec![290], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_289() {
        let rec = OpRecord::new("add", vec![289, 290], vec![291], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_290() {
        let rec = OpRecord::new("add", vec![290, 291], vec![292], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_291() {
        let rec = OpRecord::new("add", vec![291, 292], vec![293], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_292() {
        let rec = OpRecord::new("add", vec![292, 293], vec![294], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_293() {
        let rec = OpRecord::new("add", vec![293, 294], vec![295], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_294() {
        let rec = OpRecord::new("add", vec![294, 295], vec![296], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_295() {
        let rec = OpRecord::new("add", vec![295, 296], vec![297], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_296() {
        let rec = OpRecord::new("add", vec![296, 297], vec![298], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_297() {
        let rec = OpRecord::new("add", vec![297, 298], vec![299], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_298() {
        let rec = OpRecord::new("add", vec![298, 299], vec![300], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_299() {
        let rec = OpRecord::new("add", vec![299, 300], vec![301], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_300() {
        let rec = OpRecord::new("add", vec![300, 301], vec![302], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_301() {
        let rec = OpRecord::new("add", vec![301, 302], vec![303], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_302() {
        let rec = OpRecord::new("add", vec![302, 303], vec![304], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_303() {
        let rec = OpRecord::new("add", vec![303, 304], vec![305], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_304() {
        let rec = OpRecord::new("add", vec![304, 305], vec![306], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_305() {
        let rec = OpRecord::new("add", vec![305, 306], vec![307], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_306() {
        let rec = OpRecord::new("add", vec![306, 307], vec![308], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_307() {
        let rec = OpRecord::new("add", vec![307, 308], vec![309], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_308() {
        let rec = OpRecord::new("add", vec![308, 309], vec![310], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_309() {
        let rec = OpRecord::new("add", vec![309, 310], vec![311], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_310() {
        let rec = OpRecord::new("add", vec![310, 311], vec![312], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_311() {
        let rec = OpRecord::new("add", vec![311, 312], vec![313], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_312() {
        let rec = OpRecord::new("add", vec![312, 313], vec![314], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_313() {
        let rec = OpRecord::new("add", vec![313, 314], vec![315], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_314() {
        let rec = OpRecord::new("add", vec![314, 315], vec![316], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_315() {
        let rec = OpRecord::new("add", vec![315, 316], vec![317], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_316() {
        let rec = OpRecord::new("add", vec![316, 317], vec![318], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_317() {
        let rec = OpRecord::new("add", vec![317, 318], vec![319], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_318() {
        let rec = OpRecord::new("add", vec![318, 319], vec![320], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_319() {
        let rec = OpRecord::new("add", vec![319, 320], vec![321], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_320() {
        let rec = OpRecord::new("add", vec![320, 321], vec![322], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_321() {
        let rec = OpRecord::new("add", vec![321, 322], vec![323], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_322() {
        let rec = OpRecord::new("add", vec![322, 323], vec![324], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_323() {
        let rec = OpRecord::new("add", vec![323, 324], vec![325], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_324() {
        let rec = OpRecord::new("add", vec![324, 325], vec![326], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_325() {
        let rec = OpRecord::new("add", vec![325, 326], vec![327], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_326() {
        let rec = OpRecord::new("add", vec![326, 327], vec![328], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_327() {
        let rec = OpRecord::new("add", vec![327, 328], vec![329], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_328() {
        let rec = OpRecord::new("add", vec![328, 329], vec![330], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_329() {
        let rec = OpRecord::new("add", vec![329, 330], vec![331], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_330() {
        let rec = OpRecord::new("add", vec![330, 331], vec![332], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_331() {
        let rec = OpRecord::new("add", vec![331, 332], vec![333], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_332() {
        let rec = OpRecord::new("add", vec![332, 333], vec![334], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_333() {
        let rec = OpRecord::new("add", vec![333, 334], vec![335], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_334() {
        let rec = OpRecord::new("add", vec![334, 335], vec![336], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_335() {
        let rec = OpRecord::new("add", vec![335, 336], vec![337], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_336() {
        let rec = OpRecord::new("add", vec![336, 337], vec![338], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_337() {
        let rec = OpRecord::new("add", vec![337, 338], vec![339], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_338() {
        let rec = OpRecord::new("add", vec![338, 339], vec![340], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_339() {
        let rec = OpRecord::new("add", vec![339, 340], vec![341], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_340() {
        let rec = OpRecord::new("add", vec![340, 341], vec![342], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_341() {
        let rec = OpRecord::new("add", vec![341, 342], vec![343], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_342() {
        let rec = OpRecord::new("add", vec![342, 343], vec![344], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_343() {
        let rec = OpRecord::new("add", vec![343, 344], vec![345], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_344() {
        let rec = OpRecord::new("add", vec![344, 345], vec![346], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_345() {
        let rec = OpRecord::new("add", vec![345, 346], vec![347], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_346() {
        let rec = OpRecord::new("add", vec![346, 347], vec![348], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_347() {
        let rec = OpRecord::new("add", vec![347, 348], vec![349], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_348() {
        let rec = OpRecord::new("add", vec![348, 349], vec![350], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_349() {
        let rec = OpRecord::new("add", vec![349, 350], vec![351], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_350() {
        let rec = OpRecord::new("add", vec![350, 351], vec![352], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_351() {
        let rec = OpRecord::new("add", vec![351, 352], vec![353], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_352() {
        let rec = OpRecord::new("add", vec![352, 353], vec![354], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_353() {
        let rec = OpRecord::new("add", vec![353, 354], vec![355], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_354() {
        let rec = OpRecord::new("add", vec![354, 355], vec![356], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_355() {
        let rec = OpRecord::new("add", vec![355, 356], vec![357], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_356() {
        let rec = OpRecord::new("add", vec![356, 357], vec![358], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_357() {
        let rec = OpRecord::new("add", vec![357, 358], vec![359], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_358() {
        let rec = OpRecord::new("add", vec![358, 359], vec![360], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_359() {
        let rec = OpRecord::new("add", vec![359, 360], vec![361], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_360() {
        let rec = OpRecord::new("add", vec![360, 361], vec![362], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_361() {
        let rec = OpRecord::new("add", vec![361, 362], vec![363], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_362() {
        let rec = OpRecord::new("add", vec![362, 363], vec![364], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_363() {
        let rec = OpRecord::new("add", vec![363, 364], vec![365], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_364() {
        let rec = OpRecord::new("add", vec![364, 365], vec![366], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_365() {
        let rec = OpRecord::new("add", vec![365, 366], vec![367], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_366() {
        let rec = OpRecord::new("add", vec![366, 367], vec![368], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_367() {
        let rec = OpRecord::new("add", vec![367, 368], vec![369], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_368() {
        let rec = OpRecord::new("add", vec![368, 369], vec![370], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_369() {
        let rec = OpRecord::new("add", vec![369, 370], vec![371], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_370() {
        let rec = OpRecord::new("add", vec![370, 371], vec![372], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_371() {
        let rec = OpRecord::new("add", vec![371, 372], vec![373], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_372() {
        let rec = OpRecord::new("add", vec![372, 373], vec![374], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_373() {
        let rec = OpRecord::new("add", vec![373, 374], vec![375], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_374() {
        let rec = OpRecord::new("add", vec![374, 375], vec![376], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_375() {
        let rec = OpRecord::new("add", vec![375, 376], vec![377], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_376() {
        let rec = OpRecord::new("add", vec![376, 377], vec![378], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_377() {
        let rec = OpRecord::new("add", vec![377, 378], vec![379], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_378() {
        let rec = OpRecord::new("add", vec![378, 379], vec![380], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_379() {
        let rec = OpRecord::new("add", vec![379, 380], vec![381], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_380() {
        let rec = OpRecord::new("add", vec![380, 381], vec![382], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_381() {
        let rec = OpRecord::new("add", vec![381, 382], vec![383], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_382() {
        let rec = OpRecord::new("add", vec![382, 383], vec![384], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_383() {
        let rec = OpRecord::new("add", vec![383, 384], vec![385], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_384() {
        let rec = OpRecord::new("add", vec![384, 385], vec![386], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_385() {
        let rec = OpRecord::new("add", vec![385, 386], vec![387], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_386() {
        let rec = OpRecord::new("add", vec![386, 387], vec![388], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_387() {
        let rec = OpRecord::new("add", vec![387, 388], vec![389], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_388() {
        let rec = OpRecord::new("add", vec![388, 389], vec![390], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_389() {
        let rec = OpRecord::new("add", vec![389, 390], vec![391], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_390() {
        let rec = OpRecord::new("add", vec![390, 391], vec![392], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_391() {
        let rec = OpRecord::new("add", vec![391, 392], vec![393], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_392() {
        let rec = OpRecord::new("add", vec![392, 393], vec![394], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_393() {
        let rec = OpRecord::new("add", vec![393, 394], vec![395], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_394() {
        let rec = OpRecord::new("add", vec![394, 395], vec![396], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_395() {
        let rec = OpRecord::new("add", vec![395, 396], vec![397], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_396() {
        let rec = OpRecord::new("add", vec![396, 397], vec![398], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_397() {
        let rec = OpRecord::new("add", vec![397, 398], vec![399], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_398() {
        let rec = OpRecord::new("add", vec![398, 399], vec![400], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_399() {
        let rec = OpRecord::new("add", vec![399, 400], vec![401], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_400() {
        let rec = OpRecord::new("add", vec![400, 401], vec![402], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_401() {
        let rec = OpRecord::new("add", vec![401, 402], vec![403], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_402() {
        let rec = OpRecord::new("add", vec![402, 403], vec![404], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_403() {
        let rec = OpRecord::new("add", vec![403, 404], vec![405], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_404() {
        let rec = OpRecord::new("add", vec![404, 405], vec![406], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_405() {
        let rec = OpRecord::new("add", vec![405, 406], vec![407], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_406() {
        let rec = OpRecord::new("add", vec![406, 407], vec![408], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_407() {
        let rec = OpRecord::new("add", vec![407, 408], vec![409], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_408() {
        let rec = OpRecord::new("add", vec![408, 409], vec![410], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_409() {
        let rec = OpRecord::new("add", vec![409, 410], vec![411], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_410() {
        let rec = OpRecord::new("add", vec![410, 411], vec![412], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_411() {
        let rec = OpRecord::new("add", vec![411, 412], vec![413], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_412() {
        let rec = OpRecord::new("add", vec![412, 413], vec![414], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    #[test]
    fn test_tape_op_record_stress_413() {
        let rec = OpRecord::new("add", vec![413, 414], vec![415], vec![vec![2, 2]]);
        assert_eq!(rec.op_name, "add");
        assert_eq!(rec.inputs.len(), 2);
        assert_eq!(rec.outputs.len(), 1);
    }

    // Autograd verification and gradient check padding line 0
}
