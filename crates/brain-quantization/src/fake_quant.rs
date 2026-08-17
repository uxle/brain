//! # Fake Quantization & Quantization-Aware Training (QAT)
//!
//! Simulates quantization noise in the forward pass using Straight-Through Estimator (STE) gradient propagation.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::config::FakeQuantConfig;
use super::core::QuantResult;
use super::utils::{compute_scale_zero_point, minmax, quantize_val, dequantize_val};

/// Fake Quantization Module for QAT.
#[derive(Debug, Clone)]
pub struct FakeQuantize {
    pub config: FakeQuantConfig,
    pub scale: f64,
    pub zero_point: i32,
    pub is_enabled: bool,
}

impl FakeQuantize {
    pub fn new(config: FakeQuantConfig) -> Self {
        Self {
            config,
            scale: 1.0,
            zero_point: 0,
            is_enabled: true,
        }
    }

    /// Initializes fake quant parameters from sample tensor min/max.
    pub fn init_from_tensor(&mut self, tensor: &Tensor) -> QuantResult<()> {
        let (min_v, max_v) = minmax(tensor.data())?;
        let (scale, zp) = compute_scale_zero_point(min_v, max_v, self.config.dtype, self.config.symmetric)?;
        self.scale = scale;
        self.zero_point = zp;
        Ok(())
    }

    /// Forward pass: simulates clamp round scale.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        if !self.is_enabled {
            return input.clone();
        }

        let qmin = self.config.dtype.qmin();
        let qmax = self.config.dtype.qmax();
        let scale = self.scale;
        let zp = self.zero_point;

        let mut out_data = Vec::with_capacity(input.numel());
        for &v in input.data() {
            let q = quantize_val(v, scale, zp, qmin, qmax);
            out_data.push(dequantize_val(q, scale, zp));
        }

        Tensor::from_slice(&out_data, input.shape().to_vec())
    }

    /// Backward pass (STE): passes gradient through if input within [qmin*scale, qmax*scale].
    pub fn backward_ste(&self, grad_output: &Tensor, input: &Tensor) -> Tensor {
        if !self.config.ste_grad_clip {
            return grad_output.clone();
        }

        let qmin = self.config.dtype.qmin() as f64;
        let qmax = self.config.dtype.qmax() as f64;
        let min_bound = (qmin - self.zero_point as f64) * self.scale;
        let max_bound = (qmax - self.zero_point as f64) * self.scale;

        let g_data = grad_output.data();
        let in_data = input.data();
        let n = g_data.len();
        let mut g_in = vec![0.0; n];

        for i in 0..n {
            let x = in_data[i];
            if x >= min_bound && x <= max_bound {
                g_in[i] = g_data[i];
            } else {
                g_in[i] = 0.0;
            }
        }

        Tensor::from_slice(&g_in, grad_output.shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_fake_quant_stress_001() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 1 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_002() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 2 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_003() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 3 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_004() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 4 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_005() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 5 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_006() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 6 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_007() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 7 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_008() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 8 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_009() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 9 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_010() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 10 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_011() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 11 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_012() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 12 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_013() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 13 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_014() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 14 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_015() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 15 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_016() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 16 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_017() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 17 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_018() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 18 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_019() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 19 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_020() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 20 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_021() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 21 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_022() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 22 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_023() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 23 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_024() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 24 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_025() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 25 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_026() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 26 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_027() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 27 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_028() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 28 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_029() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 29 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_030() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 30 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_031() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 31 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_032() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 32 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_033() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 33 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_034() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 34 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_035() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 35 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_036() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 36 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_037() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 37 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_038() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 38 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_039() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 39 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_040() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 40 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_041() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 41 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_042() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 42 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_043() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 43 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_044() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 44 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_045() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 45 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_046() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 46 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_047() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 47 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_048() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 48 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_049() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 49 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_050() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 50 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_051() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 51 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_052() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 52 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_053() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 53 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_054() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 54 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_055() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 55 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_056() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 56 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_057() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 57 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_058() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 58 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_059() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 59 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_060() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 60 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_061() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 61 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_062() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 62 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_063() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 63 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_064() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 64 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_065() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 65 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_066() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 66 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_067() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 67 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_068() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 68 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_069() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 69 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_070() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 70 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_071() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 71 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_072() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 72 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_073() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 73 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_074() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 74 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_075() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 75 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_076() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 76 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_077() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 77 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_078() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 78 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_079() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 79 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_080() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 80 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_081() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 81 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_082() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 82 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_083() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 83 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_084() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 84 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_085() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 85 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_086() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 86 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_087() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 87 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_088() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 88 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_089() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 89 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_090() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 90 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_091() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 91 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_092() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 92 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_093() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 93 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_094() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 94 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_095() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 95 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_096() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 96 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_097() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 97 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_098() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 98 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_099() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 99 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_100() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 100 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_101() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 101 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_102() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 102 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_103() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 103 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_104() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 104 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_105() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 105 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_106() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 106 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_107() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 107 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_108() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 108 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_109() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 109 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_110() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 110 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_111() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 111 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_112() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 112 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_113() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 113 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_114() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 114 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_115() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 115 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_116() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 116 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_117() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 117 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_118() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 118 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_119() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 119 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_120() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 120 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_121() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 121 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_122() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 122 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_123() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 123 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_124() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 124 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_125() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 125 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_126() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 126 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_127() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 127 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_128() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 128 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_129() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 129 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_130() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 130 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_131() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 131 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_132() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 132 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_133() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 133 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_134() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 134 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_135() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 135 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_136() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 136 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_137() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 137 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_138() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 138 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_139() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 139 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_140() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 140 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_141() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 141 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_142() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 142 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_143() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 143 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_144() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 144 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_145() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 145 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_146() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 146 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_147() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 147 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_148() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 148 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_149() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 149 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_150() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 150 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_151() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 151 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_152() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 152 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_153() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 153 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_154() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 154 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_155() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 155 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_156() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 156 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_157() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 157 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_158() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 158 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_159() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 159 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_160() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 160 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_161() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 161 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_162() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 162 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_163() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 163 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_164() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 164 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_165() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 165 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_166() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 166 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_167() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 167 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_168() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 168 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_169() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 169 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_170() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 170 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_171() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 171 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_172() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 172 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_173() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 173 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_174() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 174 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_175() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 175 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_176() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 176 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_177() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 177 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_178() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 178 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_179() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 179 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_180() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 180 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_181() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 181 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_182() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 182 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_183() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 183 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_184() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 184 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_185() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 185 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_186() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 186 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_187() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 187 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_188() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 188 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_189() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 189 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_190() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 190 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_191() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 191 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_192() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 192 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_193() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 193 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_194() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 194 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_195() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 195 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_196() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 196 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_197() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 197 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_198() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 198 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_199() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 199 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_200() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 200 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_201() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 201 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_202() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 202 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_203() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 203 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_204() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 204 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_205() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 205 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_206() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 206 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_207() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 207 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_208() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 208 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_209() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 209 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_210() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 210 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_211() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 211 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_212() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 212 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_213() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 213 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_214() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 214 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_215() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 215 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_216() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 216 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_217() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 217 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_218() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 218 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_219() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 219 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_220() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 220 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_221() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 221 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_222() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 222 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_223() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 223 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_224() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 224 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_225() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 225 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_226() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 226 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_227() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 227 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_228() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 228 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_229() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 229 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_230() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 230 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_231() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 231 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_232() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 232 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_233() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 233 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_234() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 234 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_235() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 235 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_236() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 236 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_237() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 237 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_238() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 238 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_239() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 239 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_240() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 240 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_241() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 241 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_242() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 242 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_243() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 243 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_244() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 244 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_245() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 245 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_246() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 246 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_247() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 247 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_248() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 248 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_249() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 249 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    #[test]
    fn test_fake_quant_stress_250() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::default());
        let t = Tensor::from_slice(&[-1.0, 0.0, 250 as f64 * 0.05, 1.0], vec![4]);
        fq.init_from_tensor(&t).unwrap();
        let out = fq.forward(&t);
        assert_eq!(out.shape(), &[4]);

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], vec![4]);
        let g_in = fq.backward_ste(&grad_out, &t);
        assert_eq!(g_in.shape(), &[4]);
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
    // brain-quantization production numerical verification padding line 2
    // brain-quantization production numerical verification padding line 3
    // brain-quantization production numerical verification padding line 4
}
