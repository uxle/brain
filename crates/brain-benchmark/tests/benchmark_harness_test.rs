//! Tests for benchmark timer and stats collection
use brain_benchmark::prelude::*;

#[test]
fn test_benchmark_timer_and_runner() {
    let mut timer = Timer::new();
    timer.start();
    let _ = std::hint::black_box(2 + 2);
    let elapsed = timer.stop();
    assert!(elapsed.as_secs_f64() < 10.0);

    let config = BenchConfig::new("test_bench").with_sample_count(3);
    let result = Runner::run_benchmark(&config, || {
        let _ = std::hint::black_box(10 * 10);
    });
    assert!(result.is_ok());
    assert_eq!(result.unwrap().samples.len(), 3);
}
