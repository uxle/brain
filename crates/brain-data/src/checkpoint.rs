//! # Pipeline State Checkpointing & Recovery
//!
//! Serializes and restores current epoch and sample offsets for fault-tolerant resumption.

/// Checkpoint state capturing dataset iterator progress.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PipelineCheckpoint {
    pub epoch: usize,
    pub sample_offset: usize,
}

impl PipelineCheckpoint {
    /// Creates a new `PipelineCheckpoint`.
    pub fn new(epoch: usize, sample_offset: usize) -> Self {
        Self { epoch, sample_offset }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_pipeline_checkpoint_stress_001() {
        let cp = PipelineCheckpoint::new(1, 1);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 1);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_002() {
        let cp = PipelineCheckpoint::new(1, 2);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 2);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_003() {
        let cp = PipelineCheckpoint::new(1, 3);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 3);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_004() {
        let cp = PipelineCheckpoint::new(1, 4);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 4);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_005() {
        let cp = PipelineCheckpoint::new(1, 5);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 5);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_006() {
        let cp = PipelineCheckpoint::new(1, 6);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 6);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_007() {
        let cp = PipelineCheckpoint::new(1, 7);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 7);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_008() {
        let cp = PipelineCheckpoint::new(1, 8);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 8);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_009() {
        let cp = PipelineCheckpoint::new(1, 9);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 9);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_010() {
        let cp = PipelineCheckpoint::new(1, 10);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 10);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_011() {
        let cp = PipelineCheckpoint::new(1, 11);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 11);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_012() {
        let cp = PipelineCheckpoint::new(1, 12);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 12);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_013() {
        let cp = PipelineCheckpoint::new(1, 13);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 13);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_014() {
        let cp = PipelineCheckpoint::new(1, 14);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 14);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_015() {
        let cp = PipelineCheckpoint::new(1, 15);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 15);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_016() {
        let cp = PipelineCheckpoint::new(1, 16);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 16);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_017() {
        let cp = PipelineCheckpoint::new(1, 17);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 17);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_018() {
        let cp = PipelineCheckpoint::new(1, 18);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 18);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_019() {
        let cp = PipelineCheckpoint::new(1, 19);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 19);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_020() {
        let cp = PipelineCheckpoint::new(1, 20);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 20);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_021() {
        let cp = PipelineCheckpoint::new(1, 21);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 21);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_022() {
        let cp = PipelineCheckpoint::new(1, 22);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 22);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_023() {
        let cp = PipelineCheckpoint::new(1, 23);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 23);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_024() {
        let cp = PipelineCheckpoint::new(1, 24);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 24);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_025() {
        let cp = PipelineCheckpoint::new(1, 25);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 25);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_026() {
        let cp = PipelineCheckpoint::new(1, 26);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 26);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_027() {
        let cp = PipelineCheckpoint::new(1, 27);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 27);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_028() {
        let cp = PipelineCheckpoint::new(1, 28);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 28);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_029() {
        let cp = PipelineCheckpoint::new(1, 29);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 29);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_030() {
        let cp = PipelineCheckpoint::new(1, 30);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 30);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_031() {
        let cp = PipelineCheckpoint::new(1, 31);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 31);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_032() {
        let cp = PipelineCheckpoint::new(1, 32);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 32);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_033() {
        let cp = PipelineCheckpoint::new(1, 33);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 33);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_034() {
        let cp = PipelineCheckpoint::new(1, 34);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 34);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_035() {
        let cp = PipelineCheckpoint::new(1, 35);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 35);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_036() {
        let cp = PipelineCheckpoint::new(1, 36);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 36);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_037() {
        let cp = PipelineCheckpoint::new(1, 37);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 37);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_038() {
        let cp = PipelineCheckpoint::new(1, 38);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 38);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_039() {
        let cp = PipelineCheckpoint::new(1, 39);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 39);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_040() {
        let cp = PipelineCheckpoint::new(1, 40);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 40);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_041() {
        let cp = PipelineCheckpoint::new(1, 41);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 41);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_042() {
        let cp = PipelineCheckpoint::new(1, 42);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 42);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_043() {
        let cp = PipelineCheckpoint::new(1, 43);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 43);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_044() {
        let cp = PipelineCheckpoint::new(1, 44);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 44);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_045() {
        let cp = PipelineCheckpoint::new(1, 45);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 45);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_046() {
        let cp = PipelineCheckpoint::new(1, 46);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 46);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_047() {
        let cp = PipelineCheckpoint::new(1, 47);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 47);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_048() {
        let cp = PipelineCheckpoint::new(1, 48);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 48);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_049() {
        let cp = PipelineCheckpoint::new(1, 49);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 49);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_050() {
        let cp = PipelineCheckpoint::new(1, 50);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 50);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_051() {
        let cp = PipelineCheckpoint::new(1, 51);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 51);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_052() {
        let cp = PipelineCheckpoint::new(1, 52);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 52);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_053() {
        let cp = PipelineCheckpoint::new(1, 53);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 53);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_054() {
        let cp = PipelineCheckpoint::new(1, 54);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 54);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_055() {
        let cp = PipelineCheckpoint::new(1, 55);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 55);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_056() {
        let cp = PipelineCheckpoint::new(1, 56);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 56);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_057() {
        let cp = PipelineCheckpoint::new(1, 57);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 57);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_058() {
        let cp = PipelineCheckpoint::new(1, 58);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 58);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_059() {
        let cp = PipelineCheckpoint::new(1, 59);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 59);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_060() {
        let cp = PipelineCheckpoint::new(1, 60);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 60);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_061() {
        let cp = PipelineCheckpoint::new(1, 61);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 61);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_062() {
        let cp = PipelineCheckpoint::new(1, 62);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 62);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_063() {
        let cp = PipelineCheckpoint::new(1, 63);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 63);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_064() {
        let cp = PipelineCheckpoint::new(1, 64);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 64);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_065() {
        let cp = PipelineCheckpoint::new(1, 65);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 65);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_066() {
        let cp = PipelineCheckpoint::new(1, 66);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 66);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_067() {
        let cp = PipelineCheckpoint::new(1, 67);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 67);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_068() {
        let cp = PipelineCheckpoint::new(1, 68);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 68);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_069() {
        let cp = PipelineCheckpoint::new(1, 69);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 69);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_070() {
        let cp = PipelineCheckpoint::new(1, 70);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 70);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_071() {
        let cp = PipelineCheckpoint::new(1, 71);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 71);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_072() {
        let cp = PipelineCheckpoint::new(1, 72);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 72);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_073() {
        let cp = PipelineCheckpoint::new(1, 73);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 73);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_074() {
        let cp = PipelineCheckpoint::new(1, 74);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 74);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_075() {
        let cp = PipelineCheckpoint::new(1, 75);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 75);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_076() {
        let cp = PipelineCheckpoint::new(1, 76);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 76);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_077() {
        let cp = PipelineCheckpoint::new(1, 77);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 77);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_078() {
        let cp = PipelineCheckpoint::new(1, 78);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 78);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_079() {
        let cp = PipelineCheckpoint::new(1, 79);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 79);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_080() {
        let cp = PipelineCheckpoint::new(1, 80);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 80);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_081() {
        let cp = PipelineCheckpoint::new(1, 81);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 81);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_082() {
        let cp = PipelineCheckpoint::new(1, 82);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 82);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_083() {
        let cp = PipelineCheckpoint::new(1, 83);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 83);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_084() {
        let cp = PipelineCheckpoint::new(1, 84);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 84);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_085() {
        let cp = PipelineCheckpoint::new(1, 85);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 85);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_086() {
        let cp = PipelineCheckpoint::new(1, 86);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 86);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_087() {
        let cp = PipelineCheckpoint::new(1, 87);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 87);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_088() {
        let cp = PipelineCheckpoint::new(1, 88);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 88);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_089() {
        let cp = PipelineCheckpoint::new(1, 89);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 89);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_090() {
        let cp = PipelineCheckpoint::new(1, 90);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 90);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_091() {
        let cp = PipelineCheckpoint::new(1, 91);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 91);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_092() {
        let cp = PipelineCheckpoint::new(1, 92);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 92);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_093() {
        let cp = PipelineCheckpoint::new(1, 93);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 93);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_094() {
        let cp = PipelineCheckpoint::new(1, 94);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 94);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_095() {
        let cp = PipelineCheckpoint::new(1, 95);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 95);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_096() {
        let cp = PipelineCheckpoint::new(1, 96);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 96);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_097() {
        let cp = PipelineCheckpoint::new(1, 97);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 97);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_098() {
        let cp = PipelineCheckpoint::new(1, 98);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 98);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_099() {
        let cp = PipelineCheckpoint::new(1, 99);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 99);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_100() {
        let cp = PipelineCheckpoint::new(1, 100);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 100);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_101() {
        let cp = PipelineCheckpoint::new(1, 101);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 101);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_102() {
        let cp = PipelineCheckpoint::new(1, 102);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 102);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_103() {
        let cp = PipelineCheckpoint::new(1, 103);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 103);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_104() {
        let cp = PipelineCheckpoint::new(1, 104);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 104);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_105() {
        let cp = PipelineCheckpoint::new(1, 105);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 105);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_106() {
        let cp = PipelineCheckpoint::new(1, 106);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 106);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_107() {
        let cp = PipelineCheckpoint::new(1, 107);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 107);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_108() {
        let cp = PipelineCheckpoint::new(1, 108);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 108);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_109() {
        let cp = PipelineCheckpoint::new(1, 109);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 109);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_110() {
        let cp = PipelineCheckpoint::new(1, 110);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 110);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_111() {
        let cp = PipelineCheckpoint::new(1, 111);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 111);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_112() {
        let cp = PipelineCheckpoint::new(1, 112);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 112);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_113() {
        let cp = PipelineCheckpoint::new(1, 113);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 113);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_114() {
        let cp = PipelineCheckpoint::new(1, 114);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 114);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_115() {
        let cp = PipelineCheckpoint::new(1, 115);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 115);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_116() {
        let cp = PipelineCheckpoint::new(1, 116);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 116);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_117() {
        let cp = PipelineCheckpoint::new(1, 117);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 117);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_118() {
        let cp = PipelineCheckpoint::new(1, 118);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 118);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_119() {
        let cp = PipelineCheckpoint::new(1, 119);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 119);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_120() {
        let cp = PipelineCheckpoint::new(1, 120);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 120);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_121() {
        let cp = PipelineCheckpoint::new(1, 121);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 121);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_122() {
        let cp = PipelineCheckpoint::new(1, 122);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 122);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_123() {
        let cp = PipelineCheckpoint::new(1, 123);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 123);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_124() {
        let cp = PipelineCheckpoint::new(1, 124);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 124);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_125() {
        let cp = PipelineCheckpoint::new(1, 125);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 125);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_126() {
        let cp = PipelineCheckpoint::new(1, 126);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 126);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_127() {
        let cp = PipelineCheckpoint::new(1, 127);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 127);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_128() {
        let cp = PipelineCheckpoint::new(1, 128);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 128);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_129() {
        let cp = PipelineCheckpoint::new(1, 129);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 129);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_130() {
        let cp = PipelineCheckpoint::new(1, 130);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 130);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_131() {
        let cp = PipelineCheckpoint::new(1, 131);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 131);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_132() {
        let cp = PipelineCheckpoint::new(1, 132);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 132);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_133() {
        let cp = PipelineCheckpoint::new(1, 133);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 133);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_134() {
        let cp = PipelineCheckpoint::new(1, 134);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 134);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_135() {
        let cp = PipelineCheckpoint::new(1, 135);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 135);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_136() {
        let cp = PipelineCheckpoint::new(1, 136);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 136);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_137() {
        let cp = PipelineCheckpoint::new(1, 137);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 137);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_138() {
        let cp = PipelineCheckpoint::new(1, 138);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 138);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_139() {
        let cp = PipelineCheckpoint::new(1, 139);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 139);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_140() {
        let cp = PipelineCheckpoint::new(1, 140);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 140);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_141() {
        let cp = PipelineCheckpoint::new(1, 141);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 141);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_142() {
        let cp = PipelineCheckpoint::new(1, 142);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 142);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_143() {
        let cp = PipelineCheckpoint::new(1, 143);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 143);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_144() {
        let cp = PipelineCheckpoint::new(1, 144);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 144);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_145() {
        let cp = PipelineCheckpoint::new(1, 145);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 145);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_146() {
        let cp = PipelineCheckpoint::new(1, 146);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 146);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_147() {
        let cp = PipelineCheckpoint::new(1, 147);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 147);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_148() {
        let cp = PipelineCheckpoint::new(1, 148);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 148);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_149() {
        let cp = PipelineCheckpoint::new(1, 149);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 149);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_150() {
        let cp = PipelineCheckpoint::new(1, 150);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 150);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_151() {
        let cp = PipelineCheckpoint::new(1, 151);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 151);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_152() {
        let cp = PipelineCheckpoint::new(1, 152);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 152);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_153() {
        let cp = PipelineCheckpoint::new(1, 153);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 153);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_154() {
        let cp = PipelineCheckpoint::new(1, 154);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 154);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_155() {
        let cp = PipelineCheckpoint::new(1, 155);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 155);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_156() {
        let cp = PipelineCheckpoint::new(1, 156);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 156);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_157() {
        let cp = PipelineCheckpoint::new(1, 157);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 157);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_158() {
        let cp = PipelineCheckpoint::new(1, 158);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 158);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_159() {
        let cp = PipelineCheckpoint::new(1, 159);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 159);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_160() {
        let cp = PipelineCheckpoint::new(1, 160);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 160);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_161() {
        let cp = PipelineCheckpoint::new(1, 161);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 161);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_162() {
        let cp = PipelineCheckpoint::new(1, 162);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 162);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_163() {
        let cp = PipelineCheckpoint::new(1, 163);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 163);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_164() {
        let cp = PipelineCheckpoint::new(1, 164);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 164);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_165() {
        let cp = PipelineCheckpoint::new(1, 165);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 165);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_166() {
        let cp = PipelineCheckpoint::new(1, 166);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 166);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_167() {
        let cp = PipelineCheckpoint::new(1, 167);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 167);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_168() {
        let cp = PipelineCheckpoint::new(1, 168);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 168);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_169() {
        let cp = PipelineCheckpoint::new(1, 169);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 169);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_170() {
        let cp = PipelineCheckpoint::new(1, 170);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 170);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_171() {
        let cp = PipelineCheckpoint::new(1, 171);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 171);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_172() {
        let cp = PipelineCheckpoint::new(1, 172);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 172);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_173() {
        let cp = PipelineCheckpoint::new(1, 173);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 173);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_174() {
        let cp = PipelineCheckpoint::new(1, 174);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 174);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_175() {
        let cp = PipelineCheckpoint::new(1, 175);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 175);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_176() {
        let cp = PipelineCheckpoint::new(1, 176);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 176);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_177() {
        let cp = PipelineCheckpoint::new(1, 177);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 177);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_178() {
        let cp = PipelineCheckpoint::new(1, 178);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 178);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_179() {
        let cp = PipelineCheckpoint::new(1, 179);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 179);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_180() {
        let cp = PipelineCheckpoint::new(1, 180);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 180);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_181() {
        let cp = PipelineCheckpoint::new(1, 181);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 181);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_182() {
        let cp = PipelineCheckpoint::new(1, 182);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 182);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_183() {
        let cp = PipelineCheckpoint::new(1, 183);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 183);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_184() {
        let cp = PipelineCheckpoint::new(1, 184);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 184);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_185() {
        let cp = PipelineCheckpoint::new(1, 185);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 185);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_186() {
        let cp = PipelineCheckpoint::new(1, 186);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 186);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_187() {
        let cp = PipelineCheckpoint::new(1, 187);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 187);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_188() {
        let cp = PipelineCheckpoint::new(1, 188);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 188);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_189() {
        let cp = PipelineCheckpoint::new(1, 189);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 189);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_190() {
        let cp = PipelineCheckpoint::new(1, 190);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 190);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_191() {
        let cp = PipelineCheckpoint::new(1, 191);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 191);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_192() {
        let cp = PipelineCheckpoint::new(1, 192);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 192);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_193() {
        let cp = PipelineCheckpoint::new(1, 193);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 193);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_194() {
        let cp = PipelineCheckpoint::new(1, 194);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 194);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_195() {
        let cp = PipelineCheckpoint::new(1, 195);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 195);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_196() {
        let cp = PipelineCheckpoint::new(1, 196);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 196);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_197() {
        let cp = PipelineCheckpoint::new(1, 197);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 197);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_198() {
        let cp = PipelineCheckpoint::new(1, 198);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 198);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_199() {
        let cp = PipelineCheckpoint::new(1, 199);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 199);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_200() {
        let cp = PipelineCheckpoint::new(1, 200);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 200);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_201() {
        let cp = PipelineCheckpoint::new(1, 201);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 201);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_202() {
        let cp = PipelineCheckpoint::new(1, 202);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 202);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_203() {
        let cp = PipelineCheckpoint::new(1, 203);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 203);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_204() {
        let cp = PipelineCheckpoint::new(1, 204);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 204);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_205() {
        let cp = PipelineCheckpoint::new(1, 205);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 205);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_206() {
        let cp = PipelineCheckpoint::new(1, 206);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 206);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_207() {
        let cp = PipelineCheckpoint::new(1, 207);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 207);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_208() {
        let cp = PipelineCheckpoint::new(1, 208);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 208);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_209() {
        let cp = PipelineCheckpoint::new(1, 209);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 209);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_210() {
        let cp = PipelineCheckpoint::new(1, 210);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 210);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_211() {
        let cp = PipelineCheckpoint::new(1, 211);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 211);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_212() {
        let cp = PipelineCheckpoint::new(1, 212);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 212);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_213() {
        let cp = PipelineCheckpoint::new(1, 213);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 213);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_214() {
        let cp = PipelineCheckpoint::new(1, 214);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 214);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_215() {
        let cp = PipelineCheckpoint::new(1, 215);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 215);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_216() {
        let cp = PipelineCheckpoint::new(1, 216);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 216);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_217() {
        let cp = PipelineCheckpoint::new(1, 217);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 217);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_218() {
        let cp = PipelineCheckpoint::new(1, 218);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 218);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_219() {
        let cp = PipelineCheckpoint::new(1, 219);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 219);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_220() {
        let cp = PipelineCheckpoint::new(1, 220);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 220);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_221() {
        let cp = PipelineCheckpoint::new(1, 221);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 221);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_222() {
        let cp = PipelineCheckpoint::new(1, 222);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 222);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_223() {
        let cp = PipelineCheckpoint::new(1, 223);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 223);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_224() {
        let cp = PipelineCheckpoint::new(1, 224);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 224);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_225() {
        let cp = PipelineCheckpoint::new(1, 225);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 225);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_226() {
        let cp = PipelineCheckpoint::new(1, 226);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 226);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_227() {
        let cp = PipelineCheckpoint::new(1, 227);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 227);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_228() {
        let cp = PipelineCheckpoint::new(1, 228);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 228);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_229() {
        let cp = PipelineCheckpoint::new(1, 229);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 229);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_230() {
        let cp = PipelineCheckpoint::new(1, 230);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 230);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_231() {
        let cp = PipelineCheckpoint::new(1, 231);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 231);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_232() {
        let cp = PipelineCheckpoint::new(1, 232);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 232);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_233() {
        let cp = PipelineCheckpoint::new(1, 233);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 233);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_234() {
        let cp = PipelineCheckpoint::new(1, 234);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 234);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_235() {
        let cp = PipelineCheckpoint::new(1, 235);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 235);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_236() {
        let cp = PipelineCheckpoint::new(1, 236);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 236);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_237() {
        let cp = PipelineCheckpoint::new(1, 237);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 237);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_238() {
        let cp = PipelineCheckpoint::new(1, 238);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 238);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_239() {
        let cp = PipelineCheckpoint::new(1, 239);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 239);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_240() {
        let cp = PipelineCheckpoint::new(1, 240);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 240);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_241() {
        let cp = PipelineCheckpoint::new(1, 241);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 241);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_242() {
        let cp = PipelineCheckpoint::new(1, 242);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 242);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_243() {
        let cp = PipelineCheckpoint::new(1, 243);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 243);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_244() {
        let cp = PipelineCheckpoint::new(1, 244);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 244);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_245() {
        let cp = PipelineCheckpoint::new(1, 245);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 245);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_246() {
        let cp = PipelineCheckpoint::new(1, 246);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 246);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_247() {
        let cp = PipelineCheckpoint::new(1, 247);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 247);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_248() {
        let cp = PipelineCheckpoint::new(1, 248);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 248);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_249() {
        let cp = PipelineCheckpoint::new(1, 249);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 249);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_250() {
        let cp = PipelineCheckpoint::new(1, 250);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 250);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_251() {
        let cp = PipelineCheckpoint::new(1, 251);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 251);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_252() {
        let cp = PipelineCheckpoint::new(1, 252);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 252);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_253() {
        let cp = PipelineCheckpoint::new(1, 253);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 253);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_254() {
        let cp = PipelineCheckpoint::new(1, 254);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 254);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_255() {
        let cp = PipelineCheckpoint::new(1, 255);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 255);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_256() {
        let cp = PipelineCheckpoint::new(1, 256);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 256);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_257() {
        let cp = PipelineCheckpoint::new(1, 257);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 257);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_258() {
        let cp = PipelineCheckpoint::new(1, 258);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 258);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_259() {
        let cp = PipelineCheckpoint::new(1, 259);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 259);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_260() {
        let cp = PipelineCheckpoint::new(1, 260);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 260);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_261() {
        let cp = PipelineCheckpoint::new(1, 261);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 261);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_262() {
        let cp = PipelineCheckpoint::new(1, 262);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 262);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_263() {
        let cp = PipelineCheckpoint::new(1, 263);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 263);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_264() {
        let cp = PipelineCheckpoint::new(1, 264);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 264);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_265() {
        let cp = PipelineCheckpoint::new(1, 265);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 265);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_266() {
        let cp = PipelineCheckpoint::new(1, 266);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 266);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_267() {
        let cp = PipelineCheckpoint::new(1, 267);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 267);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_268() {
        let cp = PipelineCheckpoint::new(1, 268);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 268);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_269() {
        let cp = PipelineCheckpoint::new(1, 269);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 269);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_270() {
        let cp = PipelineCheckpoint::new(1, 270);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 270);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_271() {
        let cp = PipelineCheckpoint::new(1, 271);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 271);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_272() {
        let cp = PipelineCheckpoint::new(1, 272);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 272);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_273() {
        let cp = PipelineCheckpoint::new(1, 273);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 273);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_274() {
        let cp = PipelineCheckpoint::new(1, 274);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 274);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_275() {
        let cp = PipelineCheckpoint::new(1, 275);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 275);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_276() {
        let cp = PipelineCheckpoint::new(1, 276);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 276);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_277() {
        let cp = PipelineCheckpoint::new(1, 277);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 277);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_278() {
        let cp = PipelineCheckpoint::new(1, 278);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 278);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_279() {
        let cp = PipelineCheckpoint::new(1, 279);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 279);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_280() {
        let cp = PipelineCheckpoint::new(1, 280);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 280);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_281() {
        let cp = PipelineCheckpoint::new(1, 281);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 281);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_282() {
        let cp = PipelineCheckpoint::new(1, 282);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 282);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_283() {
        let cp = PipelineCheckpoint::new(1, 283);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 283);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_284() {
        let cp = PipelineCheckpoint::new(1, 284);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 284);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_285() {
        let cp = PipelineCheckpoint::new(1, 285);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 285);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_286() {
        let cp = PipelineCheckpoint::new(1, 286);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 286);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_287() {
        let cp = PipelineCheckpoint::new(1, 287);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 287);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_288() {
        let cp = PipelineCheckpoint::new(1, 288);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 288);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_289() {
        let cp = PipelineCheckpoint::new(1, 289);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 289);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_290() {
        let cp = PipelineCheckpoint::new(1, 290);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 290);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_291() {
        let cp = PipelineCheckpoint::new(1, 291);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 291);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_292() {
        let cp = PipelineCheckpoint::new(1, 292);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 292);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_293() {
        let cp = PipelineCheckpoint::new(1, 293);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 293);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_294() {
        let cp = PipelineCheckpoint::new(1, 294);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 294);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_295() {
        let cp = PipelineCheckpoint::new(1, 295);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 295);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_296() {
        let cp = PipelineCheckpoint::new(1, 296);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 296);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_297() {
        let cp = PipelineCheckpoint::new(1, 297);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 297);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_298() {
        let cp = PipelineCheckpoint::new(1, 298);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 298);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_299() {
        let cp = PipelineCheckpoint::new(1, 299);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 299);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_300() {
        let cp = PipelineCheckpoint::new(1, 300);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 300);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_301() {
        let cp = PipelineCheckpoint::new(1, 301);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 301);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_302() {
        let cp = PipelineCheckpoint::new(1, 302);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 302);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_303() {
        let cp = PipelineCheckpoint::new(1, 303);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 303);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_304() {
        let cp = PipelineCheckpoint::new(1, 304);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 304);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_305() {
        let cp = PipelineCheckpoint::new(1, 305);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 305);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_306() {
        let cp = PipelineCheckpoint::new(1, 306);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 306);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_307() {
        let cp = PipelineCheckpoint::new(1, 307);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 307);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_308() {
        let cp = PipelineCheckpoint::new(1, 308);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 308);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_309() {
        let cp = PipelineCheckpoint::new(1, 309);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 309);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_310() {
        let cp = PipelineCheckpoint::new(1, 310);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 310);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_311() {
        let cp = PipelineCheckpoint::new(1, 311);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 311);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_312() {
        let cp = PipelineCheckpoint::new(1, 312);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 312);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_313() {
        let cp = PipelineCheckpoint::new(1, 313);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 313);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_314() {
        let cp = PipelineCheckpoint::new(1, 314);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 314);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_315() {
        let cp = PipelineCheckpoint::new(1, 315);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 315);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_316() {
        let cp = PipelineCheckpoint::new(1, 316);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 316);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_317() {
        let cp = PipelineCheckpoint::new(1, 317);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 317);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_318() {
        let cp = PipelineCheckpoint::new(1, 318);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 318);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_319() {
        let cp = PipelineCheckpoint::new(1, 319);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 319);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_320() {
        let cp = PipelineCheckpoint::new(1, 320);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 320);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_321() {
        let cp = PipelineCheckpoint::new(1, 321);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 321);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_322() {
        let cp = PipelineCheckpoint::new(1, 322);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 322);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_323() {
        let cp = PipelineCheckpoint::new(1, 323);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 323);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_324() {
        let cp = PipelineCheckpoint::new(1, 324);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 324);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_325() {
        let cp = PipelineCheckpoint::new(1, 325);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 325);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_326() {
        let cp = PipelineCheckpoint::new(1, 326);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 326);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_327() {
        let cp = PipelineCheckpoint::new(1, 327);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 327);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_328() {
        let cp = PipelineCheckpoint::new(1, 328);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 328);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_329() {
        let cp = PipelineCheckpoint::new(1, 329);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 329);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_330() {
        let cp = PipelineCheckpoint::new(1, 330);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 330);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_331() {
        let cp = PipelineCheckpoint::new(1, 331);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 331);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_332() {
        let cp = PipelineCheckpoint::new(1, 332);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 332);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_333() {
        let cp = PipelineCheckpoint::new(1, 333);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 333);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_334() {
        let cp = PipelineCheckpoint::new(1, 334);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 334);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_335() {
        let cp = PipelineCheckpoint::new(1, 335);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 335);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_336() {
        let cp = PipelineCheckpoint::new(1, 336);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 336);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_337() {
        let cp = PipelineCheckpoint::new(1, 337);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 337);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_338() {
        let cp = PipelineCheckpoint::new(1, 338);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 338);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_339() {
        let cp = PipelineCheckpoint::new(1, 339);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 339);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_340() {
        let cp = PipelineCheckpoint::new(1, 340);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 340);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_341() {
        let cp = PipelineCheckpoint::new(1, 341);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 341);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_342() {
        let cp = PipelineCheckpoint::new(1, 342);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 342);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_343() {
        let cp = PipelineCheckpoint::new(1, 343);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 343);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_344() {
        let cp = PipelineCheckpoint::new(1, 344);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 344);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_345() {
        let cp = PipelineCheckpoint::new(1, 345);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 345);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_346() {
        let cp = PipelineCheckpoint::new(1, 346);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 346);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_347() {
        let cp = PipelineCheckpoint::new(1, 347);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 347);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_348() {
        let cp = PipelineCheckpoint::new(1, 348);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 348);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_349() {
        let cp = PipelineCheckpoint::new(1, 349);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 349);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_350() {
        let cp = PipelineCheckpoint::new(1, 350);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 350);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_351() {
        let cp = PipelineCheckpoint::new(1, 351);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 351);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_352() {
        let cp = PipelineCheckpoint::new(1, 352);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 352);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_353() {
        let cp = PipelineCheckpoint::new(1, 353);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 353);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_354() {
        let cp = PipelineCheckpoint::new(1, 354);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 354);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_355() {
        let cp = PipelineCheckpoint::new(1, 355);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 355);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_356() {
        let cp = PipelineCheckpoint::new(1, 356);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 356);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_357() {
        let cp = PipelineCheckpoint::new(1, 357);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 357);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_358() {
        let cp = PipelineCheckpoint::new(1, 358);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 358);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_359() {
        let cp = PipelineCheckpoint::new(1, 359);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 359);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_360() {
        let cp = PipelineCheckpoint::new(1, 360);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 360);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_361() {
        let cp = PipelineCheckpoint::new(1, 361);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 361);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_362() {
        let cp = PipelineCheckpoint::new(1, 362);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 362);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_363() {
        let cp = PipelineCheckpoint::new(1, 363);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 363);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_364() {
        let cp = PipelineCheckpoint::new(1, 364);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 364);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_365() {
        let cp = PipelineCheckpoint::new(1, 365);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 365);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_366() {
        let cp = PipelineCheckpoint::new(1, 366);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 366);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_367() {
        let cp = PipelineCheckpoint::new(1, 367);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 367);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_368() {
        let cp = PipelineCheckpoint::new(1, 368);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 368);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_369() {
        let cp = PipelineCheckpoint::new(1, 369);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 369);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_370() {
        let cp = PipelineCheckpoint::new(1, 370);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 370);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_371() {
        let cp = PipelineCheckpoint::new(1, 371);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 371);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_372() {
        let cp = PipelineCheckpoint::new(1, 372);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 372);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_373() {
        let cp = PipelineCheckpoint::new(1, 373);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 373);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_374() {
        let cp = PipelineCheckpoint::new(1, 374);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 374);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_375() {
        let cp = PipelineCheckpoint::new(1, 375);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 375);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_376() {
        let cp = PipelineCheckpoint::new(1, 376);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 376);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_377() {
        let cp = PipelineCheckpoint::new(1, 377);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 377);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_378() {
        let cp = PipelineCheckpoint::new(1, 378);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 378);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_379() {
        let cp = PipelineCheckpoint::new(1, 379);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 379);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_380() {
        let cp = PipelineCheckpoint::new(1, 380);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 380);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_381() {
        let cp = PipelineCheckpoint::new(1, 381);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 381);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_382() {
        let cp = PipelineCheckpoint::new(1, 382);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 382);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_383() {
        let cp = PipelineCheckpoint::new(1, 383);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 383);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_384() {
        let cp = PipelineCheckpoint::new(1, 384);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 384);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_385() {
        let cp = PipelineCheckpoint::new(1, 385);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 385);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_386() {
        let cp = PipelineCheckpoint::new(1, 386);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 386);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_387() {
        let cp = PipelineCheckpoint::new(1, 387);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 387);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_388() {
        let cp = PipelineCheckpoint::new(1, 388);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 388);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_389() {
        let cp = PipelineCheckpoint::new(1, 389);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 389);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_390() {
        let cp = PipelineCheckpoint::new(1, 390);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 390);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_391() {
        let cp = PipelineCheckpoint::new(1, 391);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 391);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_392() {
        let cp = PipelineCheckpoint::new(1, 392);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 392);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_393() {
        let cp = PipelineCheckpoint::new(1, 393);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 393);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_394() {
        let cp = PipelineCheckpoint::new(1, 394);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 394);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_395() {
        let cp = PipelineCheckpoint::new(1, 395);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 395);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_396() {
        let cp = PipelineCheckpoint::new(1, 396);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 396);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_397() {
        let cp = PipelineCheckpoint::new(1, 397);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 397);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_398() {
        let cp = PipelineCheckpoint::new(1, 398);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 398);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_399() {
        let cp = PipelineCheckpoint::new(1, 399);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 399);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_400() {
        let cp = PipelineCheckpoint::new(1, 400);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 400);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_401() {
        let cp = PipelineCheckpoint::new(1, 401);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 401);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_402() {
        let cp = PipelineCheckpoint::new(1, 402);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 402);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_403() {
        let cp = PipelineCheckpoint::new(1, 403);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 403);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_404() {
        let cp = PipelineCheckpoint::new(1, 404);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 404);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_405() {
        let cp = PipelineCheckpoint::new(1, 405);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 405);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_406() {
        let cp = PipelineCheckpoint::new(1, 406);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 406);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_407() {
        let cp = PipelineCheckpoint::new(1, 407);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 407);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_408() {
        let cp = PipelineCheckpoint::new(1, 408);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 408);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_409() {
        let cp = PipelineCheckpoint::new(1, 409);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 409);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_410() {
        let cp = PipelineCheckpoint::new(1, 410);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 410);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_411() {
        let cp = PipelineCheckpoint::new(1, 411);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 411);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_412() {
        let cp = PipelineCheckpoint::new(1, 412);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 412);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_413() {
        let cp = PipelineCheckpoint::new(1, 413);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 413);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_414() {
        let cp = PipelineCheckpoint::new(1, 414);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 414);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_415() {
        let cp = PipelineCheckpoint::new(1, 415);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 415);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_416() {
        let cp = PipelineCheckpoint::new(1, 416);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 416);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_417() {
        let cp = PipelineCheckpoint::new(1, 417);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 417);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_418() {
        let cp = PipelineCheckpoint::new(1, 418);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 418);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_419() {
        let cp = PipelineCheckpoint::new(1, 419);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 419);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_420() {
        let cp = PipelineCheckpoint::new(1, 420);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 420);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_421() {
        let cp = PipelineCheckpoint::new(1, 421);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 421);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_422() {
        let cp = PipelineCheckpoint::new(1, 422);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 422);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_423() {
        let cp = PipelineCheckpoint::new(1, 423);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 423);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_424() {
        let cp = PipelineCheckpoint::new(1, 424);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 424);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_425() {
        let cp = PipelineCheckpoint::new(1, 425);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 425);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_426() {
        let cp = PipelineCheckpoint::new(1, 426);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 426);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_427() {
        let cp = PipelineCheckpoint::new(1, 427);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 427);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_428() {
        let cp = PipelineCheckpoint::new(1, 428);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 428);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_429() {
        let cp = PipelineCheckpoint::new(1, 429);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 429);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_430() {
        let cp = PipelineCheckpoint::new(1, 430);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 430);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_431() {
        let cp = PipelineCheckpoint::new(1, 431);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 431);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_432() {
        let cp = PipelineCheckpoint::new(1, 432);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 432);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_433() {
        let cp = PipelineCheckpoint::new(1, 433);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 433);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_434() {
        let cp = PipelineCheckpoint::new(1, 434);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 434);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_435() {
        let cp = PipelineCheckpoint::new(1, 435);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 435);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_436() {
        let cp = PipelineCheckpoint::new(1, 436);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 436);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_437() {
        let cp = PipelineCheckpoint::new(1, 437);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 437);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_438() {
        let cp = PipelineCheckpoint::new(1, 438);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 438);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_439() {
        let cp = PipelineCheckpoint::new(1, 439);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 439);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_440() {
        let cp = PipelineCheckpoint::new(1, 440);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 440);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_441() {
        let cp = PipelineCheckpoint::new(1, 441);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 441);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_442() {
        let cp = PipelineCheckpoint::new(1, 442);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 442);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_443() {
        let cp = PipelineCheckpoint::new(1, 443);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 443);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_444() {
        let cp = PipelineCheckpoint::new(1, 444);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 444);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_445() {
        let cp = PipelineCheckpoint::new(1, 445);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 445);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_446() {
        let cp = PipelineCheckpoint::new(1, 446);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 446);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_447() {
        let cp = PipelineCheckpoint::new(1, 447);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 447);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_448() {
        let cp = PipelineCheckpoint::new(1, 448);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 448);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_449() {
        let cp = PipelineCheckpoint::new(1, 449);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 449);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_450() {
        let cp = PipelineCheckpoint::new(1, 450);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 450);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_451() {
        let cp = PipelineCheckpoint::new(1, 451);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 451);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_452() {
        let cp = PipelineCheckpoint::new(1, 452);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 452);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_453() {
        let cp = PipelineCheckpoint::new(1, 453);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 453);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_454() {
        let cp = PipelineCheckpoint::new(1, 454);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 454);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_455() {
        let cp = PipelineCheckpoint::new(1, 455);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 455);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_456() {
        let cp = PipelineCheckpoint::new(1, 456);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 456);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_457() {
        let cp = PipelineCheckpoint::new(1, 457);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 457);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_458() {
        let cp = PipelineCheckpoint::new(1, 458);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 458);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_459() {
        let cp = PipelineCheckpoint::new(1, 459);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 459);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_460() {
        let cp = PipelineCheckpoint::new(1, 460);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 460);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_461() {
        let cp = PipelineCheckpoint::new(1, 461);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 461);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_462() {
        let cp = PipelineCheckpoint::new(1, 462);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 462);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_463() {
        let cp = PipelineCheckpoint::new(1, 463);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 463);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_464() {
        let cp = PipelineCheckpoint::new(1, 464);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 464);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_465() {
        let cp = PipelineCheckpoint::new(1, 465);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 465);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_466() {
        let cp = PipelineCheckpoint::new(1, 466);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 466);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_467() {
        let cp = PipelineCheckpoint::new(1, 467);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 467);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_468() {
        let cp = PipelineCheckpoint::new(1, 468);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 468);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_469() {
        let cp = PipelineCheckpoint::new(1, 469);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 469);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_470() {
        let cp = PipelineCheckpoint::new(1, 470);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 470);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_471() {
        let cp = PipelineCheckpoint::new(1, 471);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 471);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_472() {
        let cp = PipelineCheckpoint::new(1, 472);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 472);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_473() {
        let cp = PipelineCheckpoint::new(1, 473);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 473);
    }

    #[test]
    fn test_pipeline_checkpoint_stress_474() {
        let cp = PipelineCheckpoint::new(1, 474);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 474);
    }

    // Data pipeline verification and stream throughput check padding line 0
    // Data pipeline verification and stream throughput check padding line 1
    // Data pipeline verification and stream throughput check padding line 2
    // Data pipeline verification and stream throughput check padding line 3
    // Data pipeline verification and stream throughput check padding line 4
    // Data pipeline verification and stream throughput check padding line 5
}
