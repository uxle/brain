# `brain-benchmark`

> Warmup-aware benchmarking, statistics, regression detection, hardware/energy profiling, and report export — all in safe Rust.

## Overview

`brain-benchmark` measures kernel, model, and I/O workloads with warmup routines, batch-size discovery, and configurable iteration strategies. Results flow through `Statistics` (percentiles, trimmed mean), `compare_runs`/`welch_t_test` for regression detection, and formatters for console, markdown, CSV, JSON, and HTML — plus Prometheus and OpenTelemetry text export.

## Features

- `Runner` with `warmup`, `discover_batch_size`, `run_benchmark`, `run_sweep`, and `bench_iter`
- `BenchConfig` builder (`with_sample_count`, `with_warmup_iterations`, `with_measurement_time`, `with_tag`, `with_flops`, `with_bytes`), `Benchmark` trait, `BenchmarkSuite` with filterable `run_all`/`run_filtered`
- `Statistics::compute` and `trimmed_mean`; `percentile_sorted`; `BenchResult` helpers (`mean_nanos`, `median_nanos`, `gflops`, `gigabytes_per_second`)
- `BaselineStore` with `evaluate_regressions`; `compare_runs` and `welch_t_test` A/B comparison
- `HardwareInfo::probe` and `estimate_memory_bandwidth_gbps`; `EnergyEstimator` (joules, GFLOPS-per-watt)
- `KernelSuite` (matmul, reduction, softmax, default suite), `ModelBenchConfig` + `bench_mlp` / `bench_transformer_layer`, `OpsBenchMatrix`
- `HdrHistogram` latency histogram, `Timer` with laps, `Profiler` event recording, `run_all` top-level entry point, `bench!`-style macros

## Modules

| Module | Description |
|---|---|
| `core` | `BenchConfig`, `Sample`, `BenchResult`, `Benchmark` trait, `MeasurementUnit`, `IterationStrategy` |
| `runner` | Warmup, batch discovery, run/sweep execution |
| `suite` | `BenchmarkSuite` collection and filtering |
| `statistics` | Mean/median/percentiles, trimmed mean, jitter |
| `baseline` | `BaselineStore` regression detection |
| `compare` | `compare_runs`, `welch_t_test`, `ComparisonResult` |
| `hardware` | `HardwareInfo::probe`, bandwidth estimation |
| `energy` | `EnergyEstimator` joules / efficiency |
| `profiler` | `Profiler` named event timing |
| `timer` | `Timer` laps and elapsed helpers |
| `histogram` | `HdrHistogram` value recording and percentiles |
| `report` | console / markdown / CSV / JSON / HTML formatters |
| `export` | Prometheus and OpenTelemetry text exporters |
| `kernels` | Matmul, reduction, softmax kernel benches |
| `models` | MLP and transformer-layer bench configs |
| `ops` | `OpsBenchMatrix` standard op matrix |
| `bench_macros` | Declarative benchmark/group macros |
| `baseline`/`graph_bench`/`io_bench`/`memory_bench`/`thread_bench`/`distribution` | Targeted workload benches |

## Quick Start

```rust
use brain_benchmark::core::BenchConfig;
use brain_benchmark::runner::Runner;

let config = BenchConfig::new("tensor_add").with_warmup_iterations(10);
let result = Runner::run_benchmark(&config, || {
    let v: Vec<f64> = (0..1_000_000).map(|x| x + 1.0).collect();
    std::hint::black_box(v.len());
}).expect("benchmark");
println!("mean: {:.2} ns", result.mean_nanos());
```

## Testing

```bash
cargo test -p brain-benchmark -j 2
```

## Workspace Role

Depends only on `brain-core`. `brain-benchmark` validates the performance claims of the framework — kernels, models, and I/O paths are benchmarked and regressions detected against saved baselines before releases.