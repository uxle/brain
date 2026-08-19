//! # Memory Allocation & Resident Set Size (RSS) Profiling
//!
//! Measures memory allocation throughput, buffer churn, and process peak RSS footprint.

use crate::core::{BenchConfig, BenchResult};
use crate::runner::Runner;
use brain_core::BrainResult;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Queries peak resident set size (VmHWM) of the current process in kilobytes.
pub fn get_peak_rss_kb() -> usize {
    if let Ok(file) = File::open("/proc/self/status") {
        let reader = BufReader::new(file);
        for line in reader.lines().map_while(Result::ok) {
            if line.starts_with("VmHWM:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        return kb;
                    }
                }
            }
        }
    }
    0
}

/// Benchmarks memory allocation and deallocation rate for dynamic vectors.
pub fn bench_vector_allocation(element_count: usize) -> BrainResult<BenchResult> {
    let bytes = (element_count * std::mem::size_of::<f64>()) as u64;
    let config = BenchConfig::new(format!("alloc_{}_elems", element_count))
        .with_bytes(bytes)
        .with_tag("memory")
        .with_tag("alloc");

    Runner::run_benchmark(&config, || {
        let v: Vec<f64> = vec![1.0; element_count];
        std::hint::black_box(v);
    })
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
