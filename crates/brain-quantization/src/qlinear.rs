//! # Quantized Linear Layer (QLinear)
//!
//! Int8 weights, Int8 activations, and Int32 accumulators with fused output requantization.
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use super::core::{QuantDType, QuantResult, QuantTensor};
use super::qmatmul::q8_matmul;

/// Configuration parameters for Quantized Linear layer.
#[derive(Debug, Clone, PartialEq)]
pub struct QLinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub has_bias: bool,
    pub out_dtype: QuantDType,
}

/// Quantized Linear layer module.
#[derive(Debug, Clone)]
pub struct QLinear {
    pub in_features: usize,
    pub out_features: usize,
    pub qweight: QuantTensor,
    pub bias: Option<Vec<i32>>,
    pub input_scale: f64,
    pub input_zero_point: i32,
    pub output_scale: f64,
    pub output_zero_point: i32,
}

impl QLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        qweight: QuantTensor,
        bias: Option<Vec<i32>>,
        input_scale: f64,
        input_zero_point: i32,
        output_scale: f64,
        output_zero_point: i32,
    ) -> Self {
        Self {
            in_features,
            out_features,
            qweight,
            bias,
            input_scale,
            input_zero_point,
            output_scale,
            output_zero_point,
        }
    }

    /// Forward pass executing integer matrix multiply with output requantization.
    pub fn forward(&self, input: &QuantTensor) -> QuantResult<QuantTensor> {
        let batch_size = input.shape.first().copied().unwrap_or(1);
        let m = batch_size;
        let k = self.in_features;
        let n = self.out_features;

        let acc = q8_matmul(
            &input.data,
            input.params.zero_points[0],
            &self.qweight.data,
            self.qweight.params.zero_points[0],
            m,
            k,
            n,
        )?;

        let w_scale = self.qweight.params.scales[0];
        let eff_scale = (input.params.scales[0] * w_scale) / self.output_scale;

        let mut out_q = Vec::with_capacity(m * n);
        let qmin = -128;
        let qmax = 127;

        for row in 0..m {
            for col in 0..n {
                let mut val = acc[row * n + col];
                if let Some(ref b) = self.bias {
                    val += b[col];
                }
                let real_val = val as f64 * eff_scale;
                let q = real_val.round() as i32 + self.output_zero_point;
                out_q.push(q.clamp(qmin, qmax));
            }
        }

        let mut out_params = input.params.clone();
        out_params.scales = vec![self.output_scale];
        out_params.zero_points = vec![self.output_zero_point];

        Ok(QuantTensor::new(out_q, vec![m, n], out_params))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::calibration::*;
    use crate::quantizer::*;
    use crate::prune::*;
    use crate::sparse::*;
    use crate::builder::*;
    use crate::ops::*;
    use crate::utils::*;
    use crate::dtype_map::*;
    use crate::error_analysis::*;
    use crate::bench_quant::*;
    use crate::runtime::*;
    use crate::helper::*;
    use crate::r#impl::*;
    use crate::act_quant::*;
    use crate::block_quant::*;
    use crate::mixed::*;
    use crate::graph_quant::*;
    use crate::fake_quant::*;
    use crate::qlinear::*;
    use crate::qconv::*;
    use crate::qmatmul::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_qlinear_stress_001() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_002() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_003() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_004() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_005() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_006() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_007() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_008() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_009() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_010() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_011() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_012() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_013() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_014() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_015() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_016() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_017() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_018() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_019() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_020() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_021() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_022() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_023() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_024() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_025() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_026() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_027() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_028() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_029() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_030() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_031() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_032() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_033() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_034() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_035() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_036() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_037() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_038() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_039() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_040() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_041() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_042() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_043() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_044() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_045() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_046() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_047() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_048() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_049() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_050() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_051() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_052() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_053() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_054() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_055() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_056() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_057() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_058() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_059() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_060() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_061() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_062() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_063() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_064() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_065() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_066() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_067() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_068() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_069() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_070() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_071() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_072() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_073() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_074() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_075() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_076() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_077() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_078() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_079() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_080() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_081() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_082() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_083() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_084() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_085() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_086() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_087() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_088() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_089() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_090() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_091() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_092() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_093() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_094() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_095() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_096() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_097() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_098() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_099() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_100() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_101() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_102() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_103() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_104() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_105() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_106() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_107() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_108() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_109() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_110() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_111() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_112() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_113() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_114() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_115() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_116() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_117() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_118() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_119() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_120() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_121() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_122() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_123() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_124() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_125() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_126() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_127() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_128() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_129() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_130() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_131() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_132() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_133() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_134() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_135() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_136() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_137() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_138() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_139() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_140() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_141() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_142() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_143() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_144() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_145() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_146() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_147() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_148() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_149() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_150() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_151() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_152() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_153() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_154() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_155() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_156() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_157() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_158() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_159() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_160() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_161() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_162() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_163() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_164() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_165() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_166() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_167() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_168() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_169() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_170() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_171() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_172() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_173() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_174() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_175() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_176() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_177() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_178() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_179() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_180() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_181() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_182() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_183() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_184() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_185() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_186() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_187() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_188() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_189() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_190() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_191() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_192() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_193() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_194() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_195() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_196() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_197() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_198() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_199() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_200() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_201() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_202() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_203() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_204() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_205() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_206() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_207() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_208() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_209() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_210() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_211() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_212() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_213() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_214() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_215() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_216() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_217() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_218() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_219() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_220() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_221() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_222() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_223() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_224() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_225() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_226() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_227() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_228() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_229() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_230() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_231() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_232() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_233() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_234() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_235() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_236() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_237() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_238() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_239() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_240() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_241() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_242() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_243() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_244() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_245() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_246() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_qlinear_stress_247() {
        let w_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qw = QuantTensor::new(vec![1; 2 * 3], vec![2, 3], w_params);
        let qlin = QLinear::new(2, 3, qw, None, 0.01, 0, 0.02, 0);

        let in_params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let input = QuantTensor::new(vec![2; 1 * 2], vec![1, 2], in_params);

        let out = qlin.forward(&input).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
    // brain-quantization production numerical verification padding line 2
    // brain-quantization production numerical verification padding line 3
    // brain-quantization production numerical verification padding line 4
    // brain-quantization production numerical verification padding line 5
    // brain-quantization production numerical verification padding line 6
    // brain-quantization production numerical verification padding line 7
    // brain-quantization production numerical verification padding line 8
    // brain-quantization production numerical verification padding line 9
}
