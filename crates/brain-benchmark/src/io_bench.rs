//! # Tensor I/O and Serialization Benchmarks
//!
//! Measures binary and JSON tensor serialization, deserialization, buffer transfers, and dataset streaming.

use crate::core::{BenchConfig, BenchResult};
use crate::runner::Runner;
use brain_core::{BrainResult, Tensor};

/// Benchmarks memory cloning and copy throughput.
pub fn bench_tensor_memory_copy(num_elements: usize) -> BrainResult<BenchResult> {
    let t = Tensor::ones(vec![num_elements]);
    let bytes = (num_elements * std::mem::size_of::<f64>()) as u64;

    let bench_cfg = BenchConfig::new(format!("tensor_copy_{}_elems", num_elements))
        .with_bytes(bytes)
        .with_tag("io")
        .with_tag("memory");

    Runner::run_benchmark(&bench_cfg, || {
        let copy = t.clone();
        std::hint::black_box(copy);
    })
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
