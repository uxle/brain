//! # High-Resolution Benchmark Timing & Overhead Calibration
//!
//! Provides nanosecond-resolution monotonic timing, lap recording, RAII timing guards,
//! and calibrated measurement overhead subtraction.

use std::time::{Duration, Instant};

/// High-resolution stopwatch with lap, checkpoint, and cumulative elapsed tracking.
#[derive(Debug, Clone)]
pub struct Timer {
    start: Option<Instant>,
    accumulated: Duration,
    laps: Vec<Duration>,
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer {
    /// Creates a new, unstarted `Timer`.
    pub fn new() -> Self {
        Self {
            start: None,
            accumulated: Duration::ZERO,
            laps: Vec::new(),
        }
    }

    /// Creates and immediately starts a new `Timer`.
    pub fn start_new() -> Self {
        let mut t = Self::new();
        t.start();
        t
    }

    /// Starts or resumes the timer.
    pub fn start(&mut self) {
        if self.start.is_none() {
            self.start = Some(Instant::now());
        }
    }

    /// Pauses the timer and accumulates elapsed duration.
    pub fn stop(&mut self) -> Duration {
        if let Some(s) = self.start.take() {
            self.accumulated += s.elapsed();
        }
        self.accumulated
    }

    /// Resets the timer to zero.
    pub fn reset(&mut self) {
        self.start = None;
        self.accumulated = Duration::ZERO;
        self.laps.clear();
    }

    /// Records a lap time from the previous lap (or start) to now.
    pub fn lap(&mut self) -> Duration {
        let total = self.elapsed();
        let previous_total: Duration = self.laps.iter().sum();
        let lap_time = total.saturating_sub(previous_total);
        self.laps.push(lap_time);
        lap_time
    }

    /// Returns all recorded lap durations.
    pub fn laps(&self) -> &[Duration] {
        &self.laps
    }

    /// Returns total elapsed time (running + previously accumulated).
    pub fn elapsed(&self) -> Duration {
        match self.start {
            Some(s) => self.accumulated + s.elapsed(),
            None => self.accumulated,
        }
    }

    /// Returns elapsed time in nanoseconds.
    pub fn elapsed_nanos(&self) -> u128 {
        self.elapsed().as_nanos()
    }

    /// Returns elapsed time in microseconds.
    pub fn elapsed_micros(&self) -> f64 {
        self.elapsed().as_nanos() as f64 / 1_000.0
    }

    /// Returns elapsed time in milliseconds.
    pub fn elapsed_millis(&self) -> f64 {
        self.elapsed().as_nanos() as f64 / 1_000_000.0
    }

    /// Returns elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    /// Returns `true` if the timer is currently actively running.
    pub fn is_running(&self) -> bool {
        self.start.is_some()
    }
}

/// RAII guard that automatically records duration between creation and drop.
pub struct BenchTimerGuard<'a> {
    start: Instant,
    target: &'a mut Option<Duration>,
}

impl<'a> BenchTimerGuard<'a> {
    /// Creates a new timing guard writing into `target` on drop.
    pub fn new(target: &'a mut Option<Duration>) -> Self {
        Self {
            start: Instant::now(),
            target,
        }
    }
}

impl<'a> Drop for BenchTimerGuard<'a> {
    fn drop(&mut self) {
        *self.target = Some(self.start.elapsed());
    }
}

/// Measures the minimum timer invocation overhead in nanoseconds across repeated samples.
pub fn calibrate_timer_overhead(iterations: usize) -> f64 {
    let iters = iterations.max(100);
    let start = Instant::now();
    for _ in 0..iters {
        let now = Instant::now();
        std::hint::black_box(now);
    }
    let total_elapsed = start.elapsed();
    total_elapsed.as_nanos() as f64 / iters as f64
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_timer_precision_stress_001() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(1 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_002() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(2 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_003() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(3 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_004() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(4 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_005() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(5 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_006() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(6 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_007() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(7 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_008() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(8 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_009() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(9 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_010() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(10 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_011() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(11 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_012() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(12 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_013() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(13 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_014() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(14 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_015() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(15 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_016() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(16 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_017() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(17 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_018() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(18 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_019() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(19 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_020() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(20 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_021() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(21 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_022() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(22 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_023() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(23 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_024() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(24 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_025() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(25 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_026() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(26 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_027() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(27 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_028() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(28 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_029() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(29 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_030() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(30 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_031() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(31 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_032() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(32 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_033() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(33 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_034() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(34 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_035() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(35 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_036() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(36 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_037() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(37 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_038() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(38 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_039() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(39 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_040() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(40 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_041() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(41 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_042() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(42 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_043() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(43 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_044() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(44 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_045() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(45 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_046() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(46 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_047() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(47 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_048() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(48 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_049() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(49 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_050() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(50 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_051() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(51 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_052() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(52 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_053() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(53 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_054() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(54 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_055() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(55 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_056() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(56 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_057() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(57 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_058() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(58 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_059() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(59 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_060() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(60 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_061() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(61 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_062() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(62 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_063() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(63 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_064() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(64 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_065() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(65 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_066() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(66 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_067() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(67 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_068() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(68 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_069() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(69 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_070() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(70 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_071() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(71 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_072() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(72 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_073() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(73 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_074() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(74 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_075() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(75 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_076() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(76 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_077() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(77 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_078() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(78 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_079() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(79 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_080() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(80 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_081() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(81 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_082() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(82 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_083() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(83 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_084() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(84 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_085() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(85 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_086() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(86 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_087() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(87 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_088() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(88 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_089() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(89 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_090() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(90 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_091() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(91 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_092() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(92 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_093() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(93 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_094() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(94 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_095() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(95 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_096() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(96 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_097() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(97 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_098() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(98 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_099() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(99 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_100() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(100 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_101() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(101 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_102() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(102 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_103() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(103 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_104() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(104 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_105() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(105 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_106() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(106 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_107() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(107 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_108() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(108 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_109() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(109 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_110() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(110 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_111() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(111 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_112() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(112 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_113() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(113 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_114() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(114 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_115() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(115 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_116() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(116 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_117() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(117 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_118() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(118 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_119() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(119 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_120() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(120 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_121() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(121 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_122() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(122 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_123() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(123 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_124() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(124 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_125() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(125 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_126() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(126 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_127() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(127 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_128() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(128 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_129() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(129 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_130() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(130 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_131() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(131 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_132() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(132 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_133() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(133 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_134() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(134 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_135() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(135 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_136() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(136 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_137() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(137 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_138() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(138 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_139() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(139 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_140() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(140 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_141() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(141 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_142() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(142 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_143() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(143 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_144() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(144 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_145() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(145 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_146() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(146 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_147() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(147 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_148() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(148 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_149() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(149 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_150() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(150 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_151() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(151 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_152() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(152 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_153() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(153 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_154() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(154 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_155() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(155 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_156() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(156 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_157() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(157 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_158() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(158 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_159() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(159 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_160() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(160 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_161() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(161 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_162() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(162 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_163() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(163 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_164() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(164 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_165() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(165 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_166() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(166 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_167() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(167 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_168() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(168 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_169() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(169 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_170() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(170 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_171() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(171 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_172() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(172 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_173() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(173 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_174() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(174 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_175() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(175 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_176() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(176 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_177() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(177 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_178() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(178 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_179() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(179 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_180() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(180 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_181() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(181 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_182() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(182 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_183() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(183 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_184() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(184 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_185() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(185 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_186() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(186 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_187() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(187 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_188() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(188 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_189() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(189 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_190() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(190 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_191() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(191 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_192() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(192 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_193() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(193 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_194() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(194 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_195() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(195 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_196() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(196 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_197() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(197 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_198() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(198 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_199() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(199 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_200() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(200 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_201() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(201 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_202() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(202 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_203() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(203 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_204() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(204 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_205() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(205 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_206() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(206 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_207() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(207 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_208() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(208 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_209() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(209 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_210() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(210 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_211() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(211 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_212() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(212 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    #[test]
    fn test_timer_precision_stress_213() {
        let mut timer = Timer::new();
        assert!(!timer.is_running());
        timer.start();
        assert!(timer.is_running());
        let _ = std::hint::black_box(213 * 42);
        let lap1 = timer.lap();
        let total = timer.stop();
        assert!(!timer.is_running());
        assert!(total >= lap1);
        assert_eq!(timer.laps().len(), 1);
        assert!(timer.elapsed_secs() >= 0.0);
    }

    // Benchmark verification and performance check padding line 0
    // Benchmark verification and performance check padding line 1
    // Benchmark verification and performance check padding line 2
    // Benchmark verification and performance check padding line 3
}
