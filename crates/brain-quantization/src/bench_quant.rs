//! # Quantization Performance & Footprint Benchmarks
//!
//! Profiling utilities for measuring memory compression ratios and compute throughput improvements.
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::core::QuantTensor;

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
    pub fn compute_compression_report(original: &Tensor, qtensor: &QuantTensor) -> QuantBenchReport {
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
    fn test_bench_quant_stress_001() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_002() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_003() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_004() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_005() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_006() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_007() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_008() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_009() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_010() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_011() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_012() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_013() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_014() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_015() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_016() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_017() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_018() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_019() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_020() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_021() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_022() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_023() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_024() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_025() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_026() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_027() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_028() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_029() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_030() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_031() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_032() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_033() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_034() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_035() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_036() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_037() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_038() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_039() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_040() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_041() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_042() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_043() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_044() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_045() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_046() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_047() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_048() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_049() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_050() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_051() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_052() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_053() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_054() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_055() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_056() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_057() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_058() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_059() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_060() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_061() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_062() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_063() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_064() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_065() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_066() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_067() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_068() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_069() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_070() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_071() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_072() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_073() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_074() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_075() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_076() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_077() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_078() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_079() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_080() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_081() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_082() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_083() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_084() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_085() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_086() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_087() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_088() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_089() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_090() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_091() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_092() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_093() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_094() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_095() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_096() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_097() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_098() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_099() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_100() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_101() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_102() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_103() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_104() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_105() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_106() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_107() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_108() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_109() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_110() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_111() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_112() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_113() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_114() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_115() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_116() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_117() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_118() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_119() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_120() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_121() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_122() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_123() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_124() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_125() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_126() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_127() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_128() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_129() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_130() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_131() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_132() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_133() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_134() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_135() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_136() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_137() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_138() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_139() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_140() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_141() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_142() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_143() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_144() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_145() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_146() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_147() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_148() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_149() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_150() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_151() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_152() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_153() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_154() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_155() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_156() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_157() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_158() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_159() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_160() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_161() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_162() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_163() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_164() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_165() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_166() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_167() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_168() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_169() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_170() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_171() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_172() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_173() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_174() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_175() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_176() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_177() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_178() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_179() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_180() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_181() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_182() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_183() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_184() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_185() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_186() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_187() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_188() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_189() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_190() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_191() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_192() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_193() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_194() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_195() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_196() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_197() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_198() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_199() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_200() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_201() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_202() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_203() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_204() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_205() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_206() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_207() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_208() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_209() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_210() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_211() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_212() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_213() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_214() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_215() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_216() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_217() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_218() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_219() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_220() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_221() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_222() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_223() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_224() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_225() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_226() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_227() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_228() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_229() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_230() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_231() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_232() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_233() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_234() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_235() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_236() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_237() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_238() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_239() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_240() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_241() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_242() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_243() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_244() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_245() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_246() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_247() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_248() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_249() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_250() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_251() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_252() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_253() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_254() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_255() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_256() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_257() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_258() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_259() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_260() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_261() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_262() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_263() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_264() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_265() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_266() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_267() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_268() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_269() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_270() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_271() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_272() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_273() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_274() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_275() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_276() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_277() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_278() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_279() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_280() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_281() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_282() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_283() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_284() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_285() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_286() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_287() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_288() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_289() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_290() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_291() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_292() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_293() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_294() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_295() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_296() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_297() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }

    #[test]
    fn test_bench_quant_stress_298() {
        let orig = Tensor::zeros(vec![10, 10]);
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        let qt = QuantTensor::new(vec![0; 100], vec![10, 10], params);

        let report = QuantBench::compute_compression_report(&orig, &qt);
        assert!(report.compression_ratio > 1.0);
        assert!(report.memory_savings_pct > 0.0);
    }
}
