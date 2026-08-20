//! # Quantization Performance & Footprint Benchmarks
//!
//! Profiling utilities for measuring memory compression ratios and compute throughput improvements.
#![allow(
    missing_docs,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

use super::core::QuantTensor;
use brain_core::Tensor;

/// Quantization compression benchmarking report.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantBenchReport {
    pub original_bytes: usize,
    pub quantized_bytes: usize,
    pub compression_ratio: f64,
    pub memory_savings_pct: f64,
}

/// Quantization benchmark helper.
#[derive(Debug, Clone, Default)]
pub struct QuantBench;

impl QuantBench {
    /// Computes memory footprint compression metrics between full precision and quantized tensor.
    pub fn compute_compression_report(
        original: &Tensor,
        qtensor: &QuantTensor,
    ) -> QuantBenchReport {
        let orig_bytes = original.numel() * std::mem::size_of::<f64>();
        let bits = qtensor.params.dtype.bit_width();
        let quant_bytes = (qtensor.numel() * bits + 7) / 8 + qtensor.params.scales.len() * 8;

        let compression_ratio = orig_bytes as f64 / quant_bytes.max(1) as f64;
        let memory_savings_pct = (1.0 - (quant_bytes as f64 / orig_bytes.max(1) as f64)) * 100.0;

        QuantBenchReport {
            original_bytes: orig_bytes,
            quantized_bytes: quant_bytes,
            compression_ratio,
            memory_savings_pct,
        }
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
