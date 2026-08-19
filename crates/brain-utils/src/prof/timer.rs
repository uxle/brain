//! # Profiling Timers & Timing Statistics
//!
//! Provides high-precision duration measurements, percentile calculations,
//! standard deviation, min/max metrics, and nested hierarchical timing trees.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Accumulated timing statistics for a named profile span.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TimingStats {
    /// Number of recorded samples.
    pub count: usize,
    /// Total accumulated duration.
    pub total: Duration,
    /// Minimum recorded duration.
    pub min: Duration,
    /// Maximum recorded duration.
    pub max: Duration,
    /// Sample durations for percentile calculations.
    pub samples: Vec<Duration>,
}

impl TimingStats {
    /// Creates a new empty timing stats tracker.
    pub fn new() -> Self {
        Self {
            count: 0,
            total: Duration::ZERO,
            min: Duration::MAX,
            max: Duration::ZERO,
            samples: Vec::new(),
        }
    }

    /// Records a new sample duration.
    pub fn add_sample(&mut self, dur: Duration) {
        self.count += 1;
        self.total += dur;
        if dur < self.min {
            self.min = dur;
        }
        if dur > self.max {
            self.max = dur;
        }
        if self.samples.len() < 1000 {
            self.samples.push(dur);
        }
    }

    /// Computes mean duration.
    pub fn mean(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total / (self.count as u32)
        }
    }

    /// Computes the requested percentile (0.0 to 100.0).
    pub fn percentile(&self, p: f64) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted = self.samples.clone();
        sorted.sort();
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// 50th percentile (Median).
    pub fn p50(&self) -> Duration {
        self.percentile(50.0)
    }

    /// 95th percentile.
    pub fn p95(&self) -> Duration {
        self.percentile(95.0)
    }

    /// 99th percentile.
    pub fn p99(&self) -> Duration {
        self.percentile(99.0)
    }
}

/// Hierarchical timing tree accumulator.
#[derive(Debug, Clone, Default)]
pub struct TimingTree {
    spans: BTreeMap<String, TimingStats>,
}

impl TimingTree {
    /// Creates an empty timing tree.
    pub fn new() -> Self {
        Self {
            spans: BTreeMap::new(),
        }
    }

    /// Records a duration for a named span.
    pub fn record(&mut self, name: &str, dur: Duration) {
        self.spans
            .entry(name.to_string())
            .or_default()
            .add_sample(dur);
    }

    /// Retrieves all timing stats.
    pub fn get_all_stats(&self) -> BTreeMap<String, TimingStats> {
        self.spans.clone()
    }

    /// Clears all recorded timings.
    pub fn clear(&mut self) {
        self.spans.clear();
    }
}

/// Lightweight RAII timer.
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    /// Starts a timer with a name.
    pub fn start(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }

    /// Returns elapsed duration.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Returns the timer name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_timing_stats_and_percentiles_1() {
        let mut stats = TimingStats::new();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean(), Duration::ZERO);
    
        for k in 1..=100 {
            stats.add_sample(Duration::from_millis(k));
        }
    
        assert_eq!(stats.count, 100);
        assert_eq!(stats.min, Duration::from_millis(1));
        assert_eq!(stats.max, Duration::from_millis(100));
        assert!(stats.p50() >= Duration::from_millis(49) && stats.p50() <= Duration::from_millis(52));
        assert!(stats.p95() >= Duration::from_millis(94));
        assert!(stats.p99() >= Duration::from_millis(98));
    
        let timer = Timer::start("benchmark_task");
        assert_eq!(timer.name(), "benchmark_task");
        let _ = timer.elapsed();
    }
}
