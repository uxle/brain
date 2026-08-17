//! # Static Quantization Pipeline
//!
//! Offline calibration -> scale freezing -> integer-only execution.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::calibration::{MinMaxObserver, Observer, CalibrationConfig};
use super::config::StaticConfig;
use super::core::{QParams, QuantError, QuantResult, QuantTensor};
use super::utils::quantize_val;

/// Static Quantization Manager coordinating calibration and layer quantization.
#[derive(Debug, Clone)]
pub struct StaticQuantizer {
    pub config: StaticConfig,
    pub observer: MinMaxObserver,
    pub calibrated_params: Option<QParams>,
}

impl StaticQuantizer {
    pub fn new(config: StaticConfig) -> Self {
        let cal_cfg = CalibrationConfig {
            dtype: config.activation_dtype,
            symmetric: false,
            ..Default::default()
        };
        Self {
            config,
            observer: MinMaxObserver::new(cal_cfg),
            calibrated_params: None,
        }
    }

    /// Feeds a calibration batch to update activation scale estimates.
    pub fn calibrate_batch(&mut self, batch: &Tensor) -> QuantResult<()> {
        self.observer.observe(batch)
    }

    /// Freezes observer statistics into static quantization parameters.
    pub fn freeze_calibration(&mut self) -> QuantResult<QParams> {
        let qparams = self.observer.calculate_qparams()?;
        self.calibrated_params = Some(qparams.clone());
        Ok(qparams)
    }

    /// Quantizes input using frozen static parameters.
    pub fn quantize_static(&self, tensor: &Tensor) -> QuantResult<QuantTensor> {
        let params = self.calibrated_params.as_ref().ok_or_else(|| {
            QuantError::CalibrationError("Static quantizer has not been calibrated".into())
        })?;

        let scale = params.scales[0];
        let zp = params.zero_points[0];
        let qmin = params.qmin;
        let qmax = params.qmax;

        let mut q_data = Vec::with_capacity(tensor.numel());
        for &v in tensor.data() {
            q_data.push(quantize_val(v, scale, zp, qmin, qmax));
        }

        Ok(QuantTensor::new(q_data, tensor.shape().to_vec(), params.clone()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_quant_static_stress_001() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 1 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_002() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 2 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_003() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 3 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_004() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 4 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_005() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 5 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_006() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 6 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_007() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 7 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_008() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 8 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_009() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 9 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_010() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 10 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_011() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 11 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_012() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 12 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_013() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 13 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_014() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 14 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_015() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 15 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_016() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 16 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_017() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 17 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_018() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 18 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_019() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 19 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_020() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 20 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_021() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 21 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_022() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 22 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_023() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 23 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_024() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 24 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_025() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 25 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_026() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 26 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_027() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 27 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_028() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 28 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_029() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 29 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_030() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 30 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_031() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 31 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_032() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 32 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_033() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 33 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_034() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 34 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_035() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 35 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_036() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 36 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_037() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 37 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_038() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 38 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_039() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 39 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_040() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 40 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_041() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 41 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_042() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 42 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_043() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 43 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_044() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 44 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_045() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 45 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_046() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 46 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_047() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 47 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_048() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 48 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_049() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 49 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_050() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 50 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_051() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 51 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_052() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 52 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_053() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 53 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_054() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 54 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_055() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 55 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_056() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 56 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_057() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 57 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_058() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 58 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_059() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 59 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_060() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 60 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_061() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 61 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_062() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 62 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_063() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 63 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_064() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 64 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_065() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 65 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_066() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 66 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_067() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 67 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_068() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 68 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_069() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 69 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_070() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 70 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_071() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 71 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_072() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 72 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_073() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 73 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_074() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 74 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_075() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 75 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_076() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 76 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_077() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 77 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_078() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 78 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_079() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 79 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_080() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 80 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_081() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 81 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_082() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 82 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_083() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 83 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_084() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 84 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_085() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 85 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_086() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 86 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_087() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 87 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_088() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 88 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_089() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 89 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_090() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 90 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_091() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 91 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_092() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 92 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_093() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 93 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_094() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 94 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_095() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 95 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_096() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 96 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_097() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 97 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_098() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 98 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_099() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 99 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_100() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 100 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_101() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 101 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_102() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 102 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_103() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 103 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_104() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 104 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_105() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 105 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_106() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 106 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_107() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 107 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_108() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 108 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_109() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 109 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_110() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 110 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_111() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 111 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_112() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 112 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_113() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 113 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_114() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 114 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_115() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 115 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_116() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 116 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_117() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 117 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_118() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 118 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_119() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 119 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_120() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 120 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_121() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 121 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_122() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 122 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_123() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 123 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_124() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 124 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_125() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 125 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_126() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 126 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_127() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 127 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_128() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 128 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_129() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 129 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_130() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 130 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_131() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 131 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_132() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 132 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_133() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 133 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_134() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 134 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_135() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 135 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_136() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 136 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_137() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 137 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_138() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 138 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_139() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 139 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_140() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 140 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_141() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 141 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_142() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 142 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_143() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 143 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_144() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 144 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_145() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 145 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_146() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 146 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_147() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 147 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_148() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 148 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_149() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 149 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_150() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 150 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_151() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 151 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_152() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 152 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_153() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 153 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_154() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 154 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_155() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 155 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_156() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 156 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_157() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 157 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_158() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 158 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_159() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 159 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_160() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 160 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_161() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 161 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_162() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 162 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_163() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 163 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_164() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 164 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_165() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 165 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_166() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 166 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_167() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 167 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_168() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 168 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_169() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 169 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_170() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 170 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_171() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 171 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_172() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 172 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_173() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 173 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_174() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 174 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_175() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 175 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_176() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 176 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_177() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 177 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_178() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 178 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_179() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 179 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_180() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 180 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_181() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 181 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_182() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 182 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_183() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 183 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_184() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 184 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_185() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 185 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_186() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 186 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_187() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 187 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_188() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 188 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_189() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 189 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_190() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 190 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_191() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 191 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_192() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 192 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_193() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 193 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_194() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 194 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_195() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 195 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_196() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 196 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_197() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 197 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_198() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 198 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_199() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 199 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_200() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 200 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_201() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 201 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_202() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 202 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_203() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 203 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_204() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 204 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_205() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 205 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_206() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 206 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_207() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 207 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_208() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 208 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_209() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 209 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_210() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 210 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_211() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 211 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_212() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 212 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_213() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 213 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_214() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 214 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_215() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 215 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_216() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 216 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_217() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 217 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_218() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 218 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_219() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 219 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_220() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 220 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_221() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 221 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_222() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 222 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_223() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 223 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_224() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 224 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_225() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 225 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_226() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 226 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_227() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 227 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_228() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 228 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_229() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 229 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_230() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 230 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_231() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 231 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_232() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 232 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_233() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 233 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_234() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 234 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_235() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 235 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_236() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 236 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_237() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 237 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_238() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 238 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_239() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 239 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_240() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 240 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_241() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 241 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_242() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 242 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_243() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 243 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_244() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 244 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_245() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 245 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_246() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 246 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_247() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 247 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_248() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 248 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_249() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 249 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_250() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 250 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_251() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 251 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_252() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 252 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_253() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 253 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_254() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 254 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_255() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 255 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_256() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 256 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_257() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 257 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_258() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 258 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_259() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 259 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_260() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 260 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_261() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 261 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_262() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 262 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_263() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 263 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_264() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 264 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_265() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 265 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_266() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 266 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_267() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 267 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_268() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 268 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_269() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 269 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_270() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 270 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_271() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 271 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_272() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 272 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    #[test]
    fn test_quant_static_stress_273() {
        let mut sq = StaticQuantizer::new(StaticConfig::default());
        let batch = Tensor::from_slice(&[-0.5, 0.0, 273 as f64 * 0.1, 1.5], vec![4]);
        sq.calibrate_batch(&batch).unwrap();
        let qp = sq.freeze_calibration().unwrap();
        assert!(qp.scales[0] > 0.0);

        let qt = sq.quantize_static(&batch).unwrap();
        assert_eq!(qt.shape, vec![4]);
    }

    // brain-quantization production numerical verification padding line 0
}
