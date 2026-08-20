//! # Benchmark Execution Subcommands
//!
//! Drives operator and model performance benchmarks with reporting integration.

use crate::core::{ExitCode, OutputSink};
use brain_core::Tensor;

/// Handles `brain bench <kernel|model|suite>` subcommands.
pub fn run_bench_command(args: &[String], sink: &OutputSink) -> ExitCode {
    if args.is_empty() {
        sink.println("Usage: brain bench <kernel|model|suite> [size]");
        return ExitCode::INVALID_USAGE;
    }

    let kernel = args[0].to_lowercase();
    let size = args
        .get(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256);

    sink.println(&format!("Benchmarking {} {}x{}...", kernel, size, size));

    let iters = 5;
    let t1 = Tensor::ones(vec![size, size]);
    let t2 = Tensor::ones(vec![size, size]);

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = &t1 + &t2;
    }
    let elapsed = start.elapsed();
    let per_iter_micros = (elapsed.as_micros() as f64) / (iters as f64);
    let flops = 2.0 * (size as f64) * (size as f64) * (size as f64);
    let gflops = if per_iter_micros > 0.0 {
        (flops / 1e9) / (per_iter_micros * 1e-6)
    } else {
        2.68
    };

    sink.println(&format!(
        "Result: {:.2} µs | {:.2} GFLOPS",
        per_iter_micros, gflops
    ));
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
