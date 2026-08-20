//! # Benchmark Execution Runner
//!
//! Drives warmup loops, adaptive sample iteration scaling, fixed duration runs,
//! and parallel benchmark workers.

use crate::core::{BenchConfig, BenchResult, IterationStrategy, Sample};
use crate::timer::Timer;
use brain_core::BrainResult;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Benchmark execution engine.
pub struct Runner;

impl Runner {
    /// Runs a warmup loop for the target closure.
    pub fn warmup<F: FnMut()>(config: &BenchConfig, f: &mut F) -> BrainResult<()> {
        let start = Instant::now();
        for _ in 0..config.warmup_iterations {
            f();
            if start.elapsed() >= config.warmup_time {
                break;
            }
        }
        Ok(())
    }

    /// Automatically discovers an appropriate iteration batch size to meet target sample duration.
    pub fn discover_batch_size<F: FnMut()>(target_duration: Duration, f: &mut F) -> usize {
        let mut iters = 1usize;
        let mut timer = Timer::new();

        loop {
            timer.reset();
            timer.start();
            for _ in 0..iters {
                f();
            }
            let elapsed = timer.stop();

            if elapsed >= target_duration || iters >= 10_000_000 {
                return iters;
            }

            let elapsed_nanos = elapsed.as_nanos().max(1) as f64;
            let target_nanos = target_duration.as_nanos() as f64;
            let ratio = (target_nanos / elapsed_nanos).max(2.0);
            iters = ((iters as f64 * ratio).ceil() as usize).max(iters + 1);
        }
    }

    /// Executes the benchmark according to `config` and records samples.
    pub fn run_benchmark<F: FnMut()>(config: &BenchConfig, mut f: F) -> BrainResult<BenchResult> {
        Self::warmup(config, &mut f)?;

        let overall_start = Instant::now();
        let mut samples = Vec::with_capacity(config.sample_count);

        let batch_size = match config.strategy {
            IterationStrategy::FixedIterations(n) => n.max(1),
            IterationStrategy::FixedDuration(d) => {
                Self::discover_batch_size(d / config.sample_count as u32, &mut f)
            }
            IterationStrategy::Adaptive {
                target_sample_duration,
                max_iterations,
            } => {
                let discovered = Self::discover_batch_size(target_sample_duration, &mut f);
                discovered.min(max_iterations).max(1)
            }
        };

        let mut timer = Timer::new();
        for _ in 0..config.sample_count {
            timer.reset();
            timer.start();
            for _ in 0..batch_size {
                f();
            }
            let elapsed = timer.stop();
            samples.push(Sample::new(elapsed, batch_size));
        }

        let total_wall = overall_start.elapsed();
        Ok(BenchResult::new(config.clone(), samples, total_wall))
    }

    /// Runs a parameter sweep across input values.
    pub fn run_sweep<P, F>(name: &str, params: &[P], mut f: F) -> BrainResult<Vec<(P, BenchResult)>>
    where
        P: Clone + std::fmt::Debug,
        F: FnMut(&P),
    {
        let mut results = Vec::new();
        for p in params {
            let config = BenchConfig::new(format!("{}_{:?}", name, p));
            let res = Self::run_benchmark(&config, || f(p))?;
            results.push((p.clone(), res));
        }
        Ok(results)
    }

    /// Runs a closure repeatedly using the `black_box` compiler fence.
    #[inline(always)]
    pub fn bench_iter<T, F: FnMut() -> T>(iters: usize, mut f: F) -> Duration {
        let start = Instant::now();
        for _ in 0..iters {
            let out = f();
            black_box(out);
        }
        start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
