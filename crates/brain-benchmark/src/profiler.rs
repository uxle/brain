//! # Lightweight Sampling & Event Profiler
//!
//! Captures timeline events, execution phases, and produces summarized latency breakdowns.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Lightweight timeline event collector.
#[derive(Debug, Clone, Default)]
pub struct Profiler {
    events: HashMap<String, Duration>,
}

impl Profiler {
    /// Creates a new `Profiler`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records duration for a named event or execution phase.
    pub fn record(&mut self, event_name: impl Into<String>, duration: Duration) {
        let entry = self
            .events
            .entry(event_name.into())
            .or_insert(Duration::ZERO);
        *entry += duration;
    }

    /// Times a closure execution and records it under `event_name`.
    pub fn time<F: FnOnce() -> R, R>(&mut self, event_name: impl Into<String>, f: F) -> R {
        let start = Instant::now();
        let res = f();
        self.record(event_name, start.elapsed());
        res
    }

    /// Returns recorded events.
    pub fn events(&self) -> &HashMap<String, Duration> {
        &self.events
    }

    /// Clears all recorded events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
