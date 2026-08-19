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

        let in_zp = input.params.zero_points.first().copied().unwrap_or(0);
        let in_scale = input.params.scales.first().copied().unwrap_or(1.0);

        let mut acc = vec![0i32; m * n];
        for j in 0..n {
            let zp_b = self.qweight.params.zero_points.get(j).copied().unwrap_or(0);
            for i in 0..m {
                for p in 0..k {
                    let a_val = input.data[i * k + p] - in_zp;
                    if a_val != 0 {
                        let b_val = self.qweight.data[p * n + j] - zp_b;
                        acc[i * n + j] = acc[i * n + j].saturating_add(a_val * b_val);
                    }
                }
            }
        }

        let mut out_q = Vec::with_capacity(m * n);
        let qmin = -128;
        let qmax = 127;

        for row in 0..m {
            for col in 0..n {
                let mut val = acc[row * n + col];
                if let Some(ref b) = self.bias {
                    val += b[col];
                }
                let w_scale = self.qweight.params.scales.get(col).copied().unwrap_or(1.0);
                let eff_scale = (in_scale * w_scale) / self.output_scale;
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
}
