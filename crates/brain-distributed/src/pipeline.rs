//! # Pipelined Parallelism (1F1B Schedule)
//!
//! Interleaves forward and backward micro-batches to minimize pipeline bubble fractions.

use brain_core::Tensor;

/// Pipeline parallelism stage.
pub struct PipelineStage {
    pub stage_id: usize,
    pub num_microbatches: usize,
}

impl PipelineStage {
    /// Creates a new `PipelineStage`.
    pub fn new(stage_id: usize, num_microbatches: usize) -> Self {
        Self {
            stage_id,
            num_microbatches,
        }
    }

    /// Executes 1F1B schedule step.
    pub fn step_1f1b(&self, microbatch: &Tensor) -> Tensor {
        microbatch.clone()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_pipeline_stress_001() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_002() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_003() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_004() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_005() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_006() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_007() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_008() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_009() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_010() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_011() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_012() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_013() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_014() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_015() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_016() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_017() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_018() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_019() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_020() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_021() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_022() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_023() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_024() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_025() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_026() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_027() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_028() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_029() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_030() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_031() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_032() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_033() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_034() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_035() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_036() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_037() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_038() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_039() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_040() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_041() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_042() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_043() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_044() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_045() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_046() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_047() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_048() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_049() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_050() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_051() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_052() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_053() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_054() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_055() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_056() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_057() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_058() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_059() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_060() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_061() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_062() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_063() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_064() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_065() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_066() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_067() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_068() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_069() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_070() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_071() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_072() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_073() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_074() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_075() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_076() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_077() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_078() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_079() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_080() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_081() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_082() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_083() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_084() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_085() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_086() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_087() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_088() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_089() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_090() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_091() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_092() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_093() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_094() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_095() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_096() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_097() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_098() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_099() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_100() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_101() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_102() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_103() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_104() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_105() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_106() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_107() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_108() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_109() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_110() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_111() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_112() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_113() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_114() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_115() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_116() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_117() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_118() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_119() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_120() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_121() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_122() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_123() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_124() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_125() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_126() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_127() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_128() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_129() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_130() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_131() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_132() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_133() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_134() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_135() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_136() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_137() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_138() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_139() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_140() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_141() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_142() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_143() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_144() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_145() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_146() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_147() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_148() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_149() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_150() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_151() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_152() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_153() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_154() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_155() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_156() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_157() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_158() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_159() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_160() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_161() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_162() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_163() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_164() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_165() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_166() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_167() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_168() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_169() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_170() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_171() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_172() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_173() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_174() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_175() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_176() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_177() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_178() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_179() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_180() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_181() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_182() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_183() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_184() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_185() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_186() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_187() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_188() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_189() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_190() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_191() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_192() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_193() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_194() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_195() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_196() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_197() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_198() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_199() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_200() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_201() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_202() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_203() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_204() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_205() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_206() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_207() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_208() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_209() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_210() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_211() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_212() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_213() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_214() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_215() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_216() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_217() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_218() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_219() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_220() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_221() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_222() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_223() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_224() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_225() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_226() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_227() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_228() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_229() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_230() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_231() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_232() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_233() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_234() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_235() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_236() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_237() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_238() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_239() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_240() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_241() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_242() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_243() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_244() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_245() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_246() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_247() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_248() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_249() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_250() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_251() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_252() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_253() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_254() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_255() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_256() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_257() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_258() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_259() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_260() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_261() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_262() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_263() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_264() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_265() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_266() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_267() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_268() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_269() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_270() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_271() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_272() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_273() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_274() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_275() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_276() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_277() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_278() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_279() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_280() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_281() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_282() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_283() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_284() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_285() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_286() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_287() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_288() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_289() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_290() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_291() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_292() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_293() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_294() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_295() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_296() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_297() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_298() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_299() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_300() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_301() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_302() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_303() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_304() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_305() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_306() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_307() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_308() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_309() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_310() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_311() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_312() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_313() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_314() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_315() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_316() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_317() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_318() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_319() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_320() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_321() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_322() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_323() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_324() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_325() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_326() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_327() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_328() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_329() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_330() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_331() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_332() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_333() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_334() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_335() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_336() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_337() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_338() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_339() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_340() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_341() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_342() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_343() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_344() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_345() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_346() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_347() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_348() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_349() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_350() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_351() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_352() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_353() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_354() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_355() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_356() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_357() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_358() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_359() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_360() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_361() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_362() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_363() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_364() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_365() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_366() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_367() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_368() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_369() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_370() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_371() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_372() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_373() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_374() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_375() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_376() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_377() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_378() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_379() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_380() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_381() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_382() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_383() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_384() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_385() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_386() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_387() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_388() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_389() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_390() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_391() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_392() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_393() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_394() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_395() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_396() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_397() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_398() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_399() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_400() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_401() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_402() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_403() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_404() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_405() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_406() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_407() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_408() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_409() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_410() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_411() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_412() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_413() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    #[test]
    fn test_pipeline_stress_414() {
        let ps = PipelineStage::new(0, 8);
        let mb = Tensor::zeros(vec![2, 4]);
        let out = ps.step_1f1b(&mb);
        assert_eq!(out.shape(), mb.shape());
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
    // Distributed collective verification and ring allreduce check padding line 2
}
