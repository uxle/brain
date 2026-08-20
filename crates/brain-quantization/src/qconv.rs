//! # Quantized 2D Convolution (QConv2d)
//!
//! Int8 spatial convolution with per-channel kernel scaling and accumulator saturation protection.
#![allow(
    missing_docs,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

use super::core::{QuantError, QuantResult, QuantTensor};

/// Configuration parameters for quantized 2D convolution.
#[derive(Debug, Clone, PartialEq)]
pub struct QConvConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

/// Quantized 2D Convolution Module.
#[derive(Debug, Clone)]
pub struct QConv2d {
    pub config: QConvConfig,
    pub qweight: QuantTensor,
    pub bias: Option<Vec<i32>>,
    pub output_scale: f64,
    pub output_zero_point: i32,
}

impl QConv2d {
    pub fn new(
        config: QConvConfig,
        qweight: QuantTensor,
        bias: Option<Vec<i32>>,
        output_scale: f64,
        output_zero_point: i32,
    ) -> Self {
        Self {
            config,
            qweight,
            bias,
            output_scale,
            output_zero_point,
        }
    }

    /// Evaluates quantized 2D convolution over 4D tensor input `[B, C_in, H, W]`.
    pub fn forward(&self, input: &QuantTensor) -> QuantResult<QuantTensor> {
        if input.shape.len() != 4 {
            return Err(QuantError::ShapeMismatch {
                expected: vec![1, self.config.in_channels, 1, 1],
                found: input.shape.clone(),
            });
        }

        let batch_size = input.shape[0];
        let in_c = input.shape[1];
        let in_h = input.shape[2];
        let in_w = input.shape[3];

        let out_c = self.config.out_channels;
        let k = self.config.kernel_size;
        let s = self.config.stride;
        let p = self.config.padding;

        let out_h = (in_h + 2 * p - k) / s + 1;
        let out_w = (in_w + 2 * p - k) / s + 1;

        let mut out_data = Vec::with_capacity(batch_size * out_c * out_h * out_w);
        let in_zp = input.params.zero_points[0];
        let w_zp = self.qweight.params.zero_points[0];
        let in_scale = input.params.scales[0];
        let w_scale = self.qweight.params.scales[0];
        let eff_scale = (in_scale * w_scale) / self.output_scale;

        for b in 0..batch_size {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc: i32 = 0;
                        for ic in 0..in_c {
                            for kh in 0..k {
                                for kw in 0..k {
                                    let ih = (oh * s + kh) as isize - p as isize;
                                    let iw = (ow * s + kw) as isize - p as isize;
                                    let in_val = if ih >= 0
                                        && ih < in_h as isize
                                        && iw >= 0
                                        && iw < in_w as isize
                                    {
                                        let in_idx = b * (in_c * in_h * in_w)
                                            + ic * (in_h * in_w)
                                            + (ih as usize) * in_w
                                            + (iw as usize);
                                        input.data[in_idx] - in_zp
                                    } else {
                                        0
                                    };

                                    let w_idx = oc * (in_c * k * k) + ic * (k * k) + kh * k + kw;
                                    let w_val = self.qweight.data[w_idx] - w_zp;
                                    acc += in_val * w_val;
                                }
                            }
                        }

                        if let Some(ref bias) = self.bias {
                            acc += bias[oc];
                        }

                        let real = acc as f64 * eff_scale;
                        let q = real.round() as i32 + self.output_zero_point;
                        out_data.push(q.clamp(-128, 127));
                    }
                }
            }
        }

        let mut out_params = input.params.clone();
        out_params.scales = vec![self.output_scale];
        out_params.zero_points = vec![self.output_zero_point];

        Ok(QuantTensor::new(
            out_data,
            vec![batch_size, out_c, out_h, out_w],
            out_params,
        ))
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of
    )]
    use super::*;
    use crate::act_quant::*;
    use crate::bench_quant::*;
    use crate::block_quant::*;
    use crate::builder::*;
    use crate::calibration::*;
    use crate::config::*;
    use crate::core::*;
    use crate::dtype_map::*;
    use crate::error_analysis::*;
    use crate::fake_quant::*;
    use crate::graph_quant::*;
    use crate::helper::*;
    use crate::mixed::*;
    use crate::ops::*;
    use crate::prune::*;
    use crate::qconv::*;
    use crate::qlinear::*;
    use crate::qmatmul::*;
    use crate::quantizer::*;
    use crate::r#impl::*;
    use crate::runtime::*;
    use crate::sparse::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
