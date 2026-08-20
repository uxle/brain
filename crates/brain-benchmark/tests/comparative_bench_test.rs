use brain_benchmark::compare::{compare_runs, ComparisonVerdict};
use brain_benchmark::core::{BenchConfig, BenchResult, Sample};
use brain_benchmark::runner::Runner;
use brain_benchmark::statistics::Statistics;
use brain_core::Tensor;
use std::time::Duration;

#[test]
fn test_tensor_matmul_benchmark_and_statistics() {
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_slice(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);

    let config = BenchConfig::new("matmul_2x2")
        .with_warmup_iterations(5)
        .with_sample_count(20);

    let result = Runner::run_benchmark(&config, || {
        let _c = brain_core::tensor::arithmetic::matmul(&a, &b);
    })
    .unwrap();

    assert_eq!(result.samples.len(), 20);
    assert!(result.mean_nanos() > 0.0);
    assert!(result.median_nanos() > 0.0);

    let stats = Statistics::compute(&result.raw_nanos);
    assert!(stats.min <= stats.median);
    assert!(stats.median <= stats.max);
    assert!(stats.p95 >= stats.p50);
    assert!(stats.p99 >= stats.p95);
}

#[test]
fn test_comparative_ab_benchmark_speedup() {
    // Simulated Baseline: 100 microseconds per iteration
    let base_samples: Vec<Sample> = vec![100_000, 102_000, 98_000, 101_000, 99_000]
        .into_iter()
        .map(|ns| Sample::new(Duration::from_nanos(ns), 1))
        .collect();
    let base_cfg = BenchConfig::new("baseline_kernel");
    let base_res = BenchResult::new(base_cfg, base_samples, Duration::from_millis(5));

    // Simulated Target (Accelerated): 25 microseconds per iteration (4x speedup)
    let target_samples: Vec<Sample> = vec![25_000, 24_000, 26_000, 25_500, 24_500]
        .into_iter()
        .map(|ns| Sample::new(Duration::from_nanos(ns), 1))
        .collect();
    let target_cfg = BenchConfig::new("target_kernel");
    let target_res = BenchResult::new(target_cfg, target_samples, Duration::from_millis(1));

    let comparison = compare_runs(&base_res, &target_res, 0.05);

    assert_eq!(comparison.verdict, ComparisonVerdict::Improvement);
    assert!(comparison.speedup_ratio >= 3.8);
    assert!(comparison.percent_change < -70.0);
    assert!(comparison.is_significant);
}
