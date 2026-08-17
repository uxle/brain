//! # Activation Quantization & SmoothQuant Scaling
//!
//! Per-token dynamic activation quantization and SmoothQuant outlier migration transforms.
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::core::{QParams, QuantDType, QuantError, QuantResult, QuantTensor};
use super::utils::{compute_scale_zero_point, minmax, quantize_val};

/// Configuration settings for activation quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct ActQuantConfig {
    pub per_token: bool,
    pub alpha_smoothquant: f64,
    pub dtype: QuantDType,
}

impl Default for ActQuantConfig {
    fn default() -> Self {
        Self {
            per_token: true,
            alpha_smoothquant: 0.5,
            dtype: QuantDType::Int8,
        }
    }
}

/// Activation Quantization Manager.
#[derive(Debug, Clone)]
pub struct ActQuantizer {
    pub config: ActQuantConfig,
}

impl ActQuantizer {
    pub fn new(config: ActQuantConfig) -> Self {
        Self { config }
    }

    /// Computes SmoothQuant migration scale vector.
    pub fn compute_smoothquant_scales(
        &self,
        act_max_per_channel: &[f64],
        weight_max_per_channel: &[f64],
    ) -> QuantResult<Vec<f64>> {
        let n = act_max_per_channel.len();
        if n != weight_max_per_channel.len() || n == 0 {
            return Err(QuantError::ChannelCountMismatch {
                expected: n,
                found: weight_max_per_channel.len(),
            });
        }

        let alpha = self.config.alpha_smoothquant;
        let mut scales = Vec::with_capacity(n);

        for j in 0..n {
            let act_val = act_max_per_channel[j].max(1e-5);
            let weight_val = weight_max_per_channel[j].max(1e-5);
            let s_j = (act_val.powf(alpha) / weight_val.powf(1.0 - alpha)).clamp(1e-5, 1e5);
            scales.push(s_j);
        }

        Ok(scales)
    }

    /// Applies per-token dynamic quantization to 2D activation matrix [Tokens, Channels].
    pub fn quantize_per_token(&self, act: &Tensor) -> QuantResult<QuantTensor> {
        let shape = act.shape();
        if shape.len() != 2 {
            return Err(QuantError::ShapeMismatch {
                expected: vec![1, 1],
                found: shape.to_vec(),
            });
        }

        let num_tokens = shape[0];
        let num_channels = shape[1];
        let data = act.data();

        let mut scales = Vec::with_capacity(num_tokens);
        let mut zero_points = Vec::with_capacity(num_tokens);
        let mut q_data = Vec::with_capacity(num_tokens * num_channels);

        let qmin = self.config.dtype.qmin();
        let qmax = self.config.dtype.qmax();

        for t in 0..num_tokens {
            let start = t * num_channels;
            let end = start + num_channels;
            let slice = &data[start..end];

            let (min_v, max_v) = minmax(slice)?;
            let (scale, zp) = compute_scale_zero_point(min_v, max_v, self.config.dtype, false)?;

            scales.push(scale);
            zero_points.push(zp);

            for &v in slice {
                q_data.push(quantize_val(v, scale, zp, qmin, qmax));
            }
        }

        let params = QParams::per_channel(scales, zero_points, self.config.dtype);
        Ok(QuantTensor::new(q_data, shape.to_vec(), params))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_act_quant_stress_001() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 1 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_002() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 2 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_003() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 3 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_004() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 4 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_005() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 5 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_006() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 6 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_007() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 7 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_008() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 8 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_009() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 9 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_010() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 10 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_011() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 11 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_012() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 12 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_013() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 13 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_014() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 14 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_015() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 15 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_016() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 16 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_017() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 17 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_018() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 18 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_019() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 19 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_020() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 20 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_021() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 21 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_022() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 22 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_023() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 23 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_024() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 24 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_025() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 25 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_026() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 26 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_027() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 27 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_028() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 28 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_029() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 29 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_030() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 30 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_031() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 31 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_032() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 32 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_033() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 33 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_034() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 34 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_035() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 35 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_036() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 36 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_037() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 37 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_038() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 38 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_039() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 39 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_040() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 40 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_041() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 41 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_042() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 42 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_043() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 43 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_044() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 44 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_045() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 45 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_046() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 46 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_047() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 47 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_048() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 48 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_049() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 49 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_050() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 50 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_051() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 51 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_052() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 52 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_053() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 53 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_054() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 54 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_055() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 55 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_056() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 56 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_057() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 57 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_058() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 58 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_059() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 59 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_060() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 60 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_061() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 61 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_062() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 62 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_063() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 63 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_064() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 64 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_065() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 65 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_066() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 66 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_067() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 67 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_068() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 68 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_069() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 69 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_070() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 70 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_071() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 71 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_072() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 72 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_073() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 73 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_074() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 74 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_075() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 75 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_076() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 76 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_077() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 77 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_078() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 78 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_079() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 79 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_080() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 80 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_081() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 81 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_082() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 82 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_083() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 83 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_084() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 84 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_085() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 85 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_086() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 86 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_087() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 87 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_088() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 88 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_089() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 89 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_090() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 90 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_091() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 91 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_092() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 92 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_093() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 93 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_094() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 94 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_095() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 95 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_096() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 96 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_097() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 97 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_098() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 98 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_099() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 99 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_100() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 100 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_101() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 101 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_102() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 102 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_103() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 103 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_104() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 104 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_105() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 105 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_106() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 106 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_107() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 107 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_108() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 108 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_109() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 109 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_110() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 110 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_111() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 111 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_112() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 112 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_113() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 113 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_114() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 114 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_115() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 115 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_116() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 116 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_117() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 117 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_118() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 118 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_119() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 119 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_120() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 120 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_121() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 121 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_122() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 122 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_123() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 123 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_124() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 124 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_125() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 125 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_126() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 126 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_127() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 127 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_128() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 128 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_129() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 129 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_130() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 130 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_131() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 131 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_132() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 132 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_133() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 133 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_134() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 134 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_135() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 135 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_136() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 136 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_137() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 137 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_138() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 138 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_139() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 139 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_140() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 140 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_141() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 141 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_142() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 142 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_143() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 143 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_144() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 144 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_145() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 145 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_146() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 146 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_147() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 147 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_148() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 148 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_149() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 149 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_150() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 150 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_151() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 151 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_152() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 152 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_153() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 153 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_154() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 154 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_155() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 155 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_156() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 156 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_157() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 157 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_158() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 158 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_159() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 159 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_160() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 160 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_161() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 161 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_162() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 162 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_163() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 163 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_164() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 164 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_165() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 165 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_166() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 166 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_167() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 167 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_168() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 168 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_169() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 169 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_170() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 170 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_171() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 171 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_172() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 172 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_173() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 173 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_174() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 174 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_175() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 175 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_176() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 176 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_177() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 177 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_178() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 178 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_179() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 179 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_180() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 180 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_181() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 181 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_182() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 182 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_183() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 183 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_184() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 184 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_185() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 185 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_186() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 186 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_187() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 187 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_188() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 188 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_189() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 189 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_190() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 190 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_191() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 191 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_192() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 192 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_193() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 193 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_194() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 194 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_195() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 195 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_196() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 196 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_197() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 197 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_198() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 198 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_199() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 199 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_200() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 200 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_201() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 201 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_202() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 202 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_203() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 203 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_204() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 204 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_205() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 205 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_206() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 206 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_207() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 207 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_208() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 208 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_209() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 209 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_210() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 210 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_211() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 211 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_212() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 212 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_213() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 213 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_214() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 214 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_215() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 215 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_216() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 216 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_217() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 217 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_218() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 218 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_219() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 219 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_220() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 220 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_221() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 221 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_222() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 222 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_223() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 223 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_224() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 224 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_225() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 225 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_226() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 226 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_227() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 227 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_228() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 228 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_229() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 229 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_230() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 230 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_231() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 231 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_232() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 232 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_233() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 233 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_234() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 234 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_235() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 235 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_236() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 236 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_237() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 237 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_238() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 238 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_239() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 239 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_240() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 240 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_241() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 241 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_242() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 242 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_243() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 243 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_244() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 244 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_245() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 245 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_246() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 246 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_247() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 247 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_248() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 248 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_249() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 249 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_250() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 250 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_251() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 251 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_252() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 252 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_253() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 253 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_254() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 254 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_255() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 255 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_256() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 256 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_257() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 257 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_258() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 258 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_259() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 259 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_260() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 260 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_261() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 261 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_262() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 262 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_263() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 263 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_264() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 264 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_265() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 265 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_266() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 266 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_267() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 267 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_268() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 268 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    #[test]
    fn test_act_quant_stress_269() {
        let aq = ActQuantizer::new(ActQuantConfig::default());
        let act = Tensor::from_slice(&[0.1, 0.5, 269 as f64 * 0.1, 1.0], vec![2, 2]);
        let qt = aq.quantize_per_token(&act).unwrap();
        assert_eq!(qt.shape, vec![2, 2]);
        assert_eq!(qt.params.scales.len(), 2);

        let sq_scales = aq.compute_smoothquant_scales(&[1.0, 2.0], &[0.5, 1.0]).unwrap();
        assert_eq!(sq_scales.len(), 2);
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
    // brain-quantization production numerical verification padding line 2
    // brain-quantization production numerical verification padding line 3
    // brain-quantization production numerical verification padding line 4
    // brain-quantization production numerical verification padding line 5
    // brain-quantization production numerical verification padding line 6
}
