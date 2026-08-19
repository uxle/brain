//! # Reference Kernel Benchmarks
//!
//! Benchmark implementations for fundamental compute kernels in `brain-core`:
//! Matrix Multiplication, 2D Convolution, Reductions, Softmax, and Elementwise Pipelines.

use crate::core::{BenchConfig, BenchResult};
use crate::runner::Runner;
use brain_core::tensor::arithmetic as arith_t;
use brain_core::tensor::reduction as red_t;
use brain_core::tensor::special as spec_t;
use brain_core::{BrainResult, Tensor};

/// Suite of standard mathematical kernels.
pub struct KernelSuite;

impl KernelSuite {
    /// Benchmarks Matrix Multiplication `[M, K] x [K, N] -> [M, N]` with FLOP calculation.
    pub fn bench_matmul(m: usize, k: usize, n: usize) -> BrainResult<BenchResult> {
        let a = Tensor::ones(vec![m, k]);
        let b = Tensor::ones(vec![k, n]);
        let flops = 2 * (m as u64) * (k as u64) * (n as u64);

        let config = BenchConfig::new(format!("matmul_{}x{}x{}", m, k, n))
            .with_flops(flops)
            .with_tag("kernel")
            .with_tag("matmul");

        Runner::run_benchmark(&config, || {
            let c = arith_t::matmul(&a, &b);
            std::hint::black_box(c);
        })
    }

    /// Benchmarks Sum Reduction over a 1D tensor buffer.
    pub fn bench_reduction(num_elements: usize) -> BrainResult<BenchResult> {
        let a = Tensor::ones(vec![num_elements]);
        let bytes = (num_elements * std::mem::size_of::<f64>()) as u64;

        let config = BenchConfig::new(format!("reduction_sum_{}", num_elements))
            .with_bytes(bytes)
            .with_tag("kernel")
            .with_tag("reduction");

        Runner::run_benchmark(&config, || {
            let s = red_t::sum(&a);
            std::hint::black_box(s);
        })
    }

    /// Benchmarks Softmax over batch dimension.
    pub fn bench_softmax(batch: usize, dim: usize) -> BrainResult<BenchResult> {
        let a = Tensor::ones(vec![batch, dim]);
        let config = BenchConfig::new(format!("softmax_{}x{}", batch, dim))
            .with_tag("kernel")
            .with_tag("special");

        Runner::run_benchmark(&config, || {
            let s = spec_t::softmax(&a, 1);
            std::hint::black_box(s);
        })
    }

    /// Runs all standard default reference kernels.
    pub fn default_suite() -> BrainResult<Vec<BenchResult>> {
        Ok(vec![
            Self::bench_matmul(64, 64, 64)?,
            Self::bench_reduction(10_000)?,
            Self::bench_softmax(32, 128)?,
        ])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
