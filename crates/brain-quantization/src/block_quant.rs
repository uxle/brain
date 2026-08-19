//! # Block and Group-wise Quantization
//!
//! GPTQ & AWQ style fine-grained sub-tensor group scaling (group sizes 32, 64, 128) for LLM compression.
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::config::BlockQuantConfig;
use super::core::{QParams, QuantResult, QuantScheme, QuantTensor};
use super::utils::{compute_scale_zero_point, minmax, quantize_val};

/// Block/Group-wise Quantizer.
#[derive(Debug, Clone)]
pub struct BlockQuantizer {
    pub config: BlockQuantConfig,
}

impl BlockQuantizer {
    pub fn new(config: BlockQuantConfig) -> Self {
        Self { config }
    }

    /// Quantizes tensor weights in contiguous blocks of size `group_size`.
    pub fn quantize_blocks(&self, tensor: &Tensor) -> QuantResult<QuantTensor> {
        let total_elements = tensor.numel();
        let group_size = self.config.group_size.max(1);
        let num_groups = (total_elements + group_size - 1) / group_size;

        let data = tensor.data();
        let mut scales = Vec::with_capacity(num_groups);
        let mut zero_points = Vec::with_capacity(num_groups);
        let mut q_data = Vec::with_capacity(total_elements);

        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(total_elements);
            let slice = &data[start..end];

            let (min_v, max_v) = minmax(slice)?;
            let (scale, zp) = compute_scale_zero_point(min_v, max_v, self.config.dtype, self.config.symmetric)?;

            scales.push(scale);
            zero_points.push(zp);

            let qmin = self.config.dtype.qmin();
            let qmax = self.config.dtype.qmax();

            for &v in slice {
                q_data.push(quantize_val(v, scale, zp, qmin, qmax));
            }
        }

        let params = QParams {
            scales,
            zero_points,
            qmin: self.config.dtype.qmin(),
            qmax: self.config.dtype.qmax(),
            scheme: QuantScheme::GroupWise { group_size },
            dtype: self.config.dtype,
        };

        Ok(QuantTensor::new(q_data, tensor.shape().to_vec(), params))
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
