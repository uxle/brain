//! # Profiling Framework
//!
//! Production-grade profiling infrastructure providing scoped timers,
//! call trees, memory counters, flame graph exports, and profiling sessions.

pub mod counters;
pub mod scope;
pub mod timer;

use self::counters::CounterSet;
use self::timer::{TimingStats, TimingTree};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

/// Profiler configuration options.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfConfig {
    /// Whether profiling is globally enabled.
    pub enabled: bool,
    /// Minimum span duration in microseconds to record.
    pub min_duration_us: u64,
    /// Maximum call-tree depth.
    pub max_depth: usize,
    /// Whether to track memory allocation stats.
    pub track_memory: bool,
}

impl Default for ProfConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_duration_us: 1,
            max_depth: 32,
            track_memory: false,
        }
    }
}

/// Central profiler managing timing trees and metric counters.
pub struct Profiler {
    config: RwLock<ProfConfig>,
    tree: Arc<Mutex<TimingTree>>,
    counters: Arc<CounterSet>,
}

impl Profiler {
    /// Constructs a new profiler.
    pub fn new(config: ProfConfig) -> Self {
        Self {
            config: RwLock::new(config),
            tree: Arc::new(Mutex::new(TimingTree::new())),
            counters: Arc::new(CounterSet::new()),
        }
    }

    /// Default profiler instance.
    pub fn default_profiler() -> Self {
        Self::new(ProfConfig::default())
    }

    /// Records a completed execution span.
    pub fn record_span(&self, name: &str, duration: Duration) {
        let enabled = self.config.read().map(|c| c.enabled).unwrap_or(false);
        if enabled {
            if let Ok(mut t) = self.tree.lock() {
                t.record(name, duration);
            }
        }
    }

    /// Gets summary statistics for all spans.
    pub fn get_stats(&self) -> BTreeMap<String, TimingStats> {
        self.tree
            .lock()
            .map(|t| t.get_all_stats())
            .unwrap_or_default()
    }

    /// Returns a reference to the counter set.
    pub fn counters(&self) -> &Arc<CounterSet> {
        &self.counters
    }

    /// Clears recorded profiles.
    pub fn reset(&self) {
        if let Ok(mut t) = self.tree.lock() {
            t.clear();
        }
        self.counters.reset_all();
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_profiler_lifecycle_1() {
        let prof = Profiler::default_profiler();
        prof.record_span("forward_pass", Duration::from_micros(150));
        prof.record_span("backward_pass", Duration::from_micros(300));

        let stats = prof.get_stats();
        assert!(stats.contains_key("forward_pass"));
        assert!(stats.contains_key("backward_pass"));
        assert_eq!(stats.get("forward_pass").unwrap().count, 1);

        prof.counters().get_counter("batches_processed").inc();
        assert_eq!(prof.counters().get_counter("batches_processed").get(), 1);

        prof.reset();
        let reset_stats = prof.get_stats();
        assert!(reset_stats.is_empty());
    }
}
