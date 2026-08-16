# `brain-benchmark` (v0.2.0)

> Benchmarking, Memory Profiling, Statistical Metric Analysis, and Throughput Measurement Suite.

## Overview

`brain-benchmark` is a dedicated benchmarking and performance analysis crate for neural network models and tensor operations. It provides warmup routines, high-resolution timers, statistical summarization (mean, median, p95, p99, stddev), FLOP estimation, memory allocation tracking, and latency regression detection.

## Architecture

| Module | Description |
|---|---|
| `timer` | High-precision wall-clock and CPU cycle measurement tools |
| `stats` | Statistical aggregators: percentiles (p50, p90, p95, p99), jitter, confidence intervals |
| `profiler` | Layer-wise time and memory allocation profiling |
| `flops` | Theoretical and empirical FLOP count estimation for conv, linear, attention layers |
| `reporter` | Formatted markdown, CSV, and terminal report generators |

## Quick Start

```rust
use brain_benchmark::timer::Benchmark;

fn main() {
    let mut bench = Benchmark::new("Tensor Add 1M elements");
    let report = bench.run(100, || {
        let v: Vec<f64> = (0..1_000_000).map(|x| x as f64 + 1.0).collect();
        v.len()
    });
    println!("{}", report.summary());
}
```

## Quality & Verification

- **Tests**: 8,421 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-benchmark -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
