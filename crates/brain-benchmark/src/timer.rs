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
}
