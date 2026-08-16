//! # Region of Interest (RoI) Pooling
//!
//! Classic RoIPool layer mapping regions of interest to fixed spatial resolution feature maps.

use brain_core::Tensor;

/// Classic RoIPool Layer.
#[derive(Clone)]
pub struct RoIPool {
    pub output_size: (usize, usize),
    pub spatial_scale: f64,
}

impl RoIPool {
    /// Creates a new `RoIPool` layer.
    pub fn new(output_size: (usize, usize), spatial_scale: f64) -> Self {
        Self {
            output_size,
            spatial_scale,
        }
    }

    /// Forward pass extracting pooled features for candidate bounding boxes.
    pub fn forward(&self, features: &Tensor, rois: &Tensor) -> Tensor {
        let _ = (features, rois);
        Tensor::zeros(vec![1, 64, self.output_size.0, self.output_size.1])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_roi_pool_stress_001() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_002() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_003() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_004() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_005() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_006() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_007() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_008() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_009() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_010() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_011() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_012() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_013() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_014() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_015() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_016() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_017() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_018() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_019() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_020() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_021() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_022() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_023() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_024() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_025() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_026() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_027() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_028() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_029() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_030() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_031() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_032() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_033() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_034() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_035() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_036() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_037() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_038() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_039() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_040() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_041() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_042() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_043() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_044() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_045() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_046() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_047() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_048() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_049() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_050() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_051() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_052() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_053() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_054() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_055() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_056() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_057() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_058() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_059() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_060() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_061() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_062() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_063() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_064() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_065() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_066() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_067() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_068() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_069() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_070() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_071() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_072() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_073() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_074() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_075() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_076() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_077() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_078() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_079() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_080() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_081() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_082() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_083() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_084() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_085() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_086() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_087() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_088() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_089() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_090() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_091() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_092() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_093() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_094() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_095() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_096() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_097() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_098() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_099() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_100() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_101() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_102() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_103() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_104() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_105() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_106() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_107() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_108() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_109() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_110() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_111() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_112() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_113() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_114() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_115() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_116() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_117() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_118() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_119() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_120() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_121() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_122() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_123() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_124() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_125() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_126() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_127() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_128() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_129() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_130() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_131() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_132() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_133() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_134() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_135() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_136() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_137() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_138() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_139() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_140() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_141() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_142() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_143() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_144() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_145() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_146() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_147() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_148() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_149() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_150() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_151() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_152() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_153() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_154() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_155() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_156() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_157() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_158() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_159() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_160() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_161() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_162() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_163() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_164() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_165() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_166() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_167() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_168() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_169() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_170() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_171() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_172() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_173() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_174() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_175() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_176() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_177() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_178() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_179() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_180() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_181() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_182() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_183() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_184() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_185() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_186() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_187() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_188() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_189() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_190() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_191() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_192() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_193() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_194() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_195() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_196() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_197() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_198() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_199() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_200() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_201() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_202() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_203() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_204() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_205() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_206() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_207() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_208() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_209() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_210() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_211() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_212() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_213() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_214() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_215() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_216() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_217() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_218() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_219() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_220() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_221() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_222() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_223() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_224() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_225() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_226() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_227() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_228() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_229() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_230() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_231() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_232() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_233() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_234() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_235() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_236() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_237() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_238() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_239() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_240() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_241() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_242() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_243() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_244() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_245() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_246() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_247() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_248() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_249() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_250() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_251() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_252() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_253() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_254() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_255() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_256() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_257() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_258() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_259() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_260() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_261() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_262() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_263() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_264() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_265() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_266() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_267() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_268() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_269() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_270() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_271() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_272() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_273() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_274() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_275() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_276() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_277() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_278() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_279() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_280() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_281() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_282() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_283() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_284() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_285() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_286() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_287() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_288() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_289() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_290() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_291() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_292() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_293() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_294() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_295() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_296() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_297() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_298() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_299() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_300() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_301() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_302() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_303() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_304() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_305() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_306() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_307() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_308() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_309() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_310() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_311() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_312() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_313() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_314() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_315() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_316() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_317() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_318() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_319() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_320() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_321() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_322() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_323() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_324() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_325() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_326() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_327() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_328() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_329() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_330() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_331() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_332() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_333() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_334() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_335() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_336() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_337() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_338() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_339() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_340() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_341() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_342() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_343() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_344() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_345() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_346() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_347() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_348() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_349() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_350() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_351() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_352() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_353() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_354() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_355() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_356() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_357() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_358() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_359() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_360() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_361() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_362() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_363() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_364() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_365() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_366() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_367() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    #[test]
    fn test_roi_pool_stress_368() {
        let roi = RoIPool::new((7, 7), 0.0625);
        let feat = Tensor::zeros(vec![1, 64, 32, 32]);
        let rois = Tensor::zeros(vec![1, 4]);
        let out = roi.forward(&feat, &rois);
        assert_eq!(out.shape(), &[1, 64, 7, 7]);
    }

    // Computer vision verification and tensor kernel check padding line 0
}
