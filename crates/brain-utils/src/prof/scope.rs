//! # Profiling Scope Guards
//!
//! Provides RAII scope guards for automatic span timing and hierarchical
//! profiling report generation.

use std::time::{Duration, Instant};

/// Scoped timing guard that captures execution duration upon drop.
pub struct ScopeGuard<'a, F: FnMut(&str, Duration)> {
    name: &'a str,
    start: Instant,
    on_drop: F,
}

impl<'a, F: FnMut(&str, Duration)> ScopeGuard<'a, F> {
    /// Creates a new scope guard.
    pub fn new(name: &'a str, on_drop: F) -> Self {
        Self {
            name,
            start: Instant::now(),
            on_drop,
        }
    }

    /// Elapsed time since scope entry.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl<'a, F: FnMut(&str, Duration)> Drop for ScopeGuard<'a, F> {
    fn drop(&mut self) {
        let dur = self.start.elapsed();
        (self.on_drop)(self.name, dur);
    }
}

/// A structured report of captured execution scopes.
#[derive(Debug, Clone, PartialEq)]
pub struct ScopeReport {
    /// Name of top-level scope.
    pub name: String,
    /// Total duration spent in scope.
    pub total_duration: Duration,
    /// Child scope breakdown.
    pub children: Vec<ScopeReport>,
}

impl ScopeReport {
    /// Creates a new scope report.
    pub fn new(name: &str, duration: Duration) -> Self {
        Self {
            name: name.to_string(),
            total_duration: duration,
            children: Vec::new(),
        }
    }

    /// Adds a child sub-scope.
    pub fn add_child(&mut self, child: ScopeReport) -> &mut Self {
        self.children.push(child);
        self
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_scope_guard_raii_1() {
        let mut recorded_name = String::new();
        let mut recorded_dur = Duration::ZERO;
    
        {
            let _guard = ScopeGuard::new("matrix_multiply", |name, dur| {
                recorded_name = name.to_string();
                recorded_dur = dur;
            });
            assert_eq!(_guard.name, "matrix_multiply");
            let _ = _guard.elapsed();
        }
    
        assert_eq!(recorded_name, "matrix_multiply");
        let _ = recorded_dur;
    
        let mut report = ScopeReport::new("epoch_1", Duration::from_millis(500));
        report.add_child(ScopeReport::new("dataloader", Duration::from_millis(100)));
        report.add_child(ScopeReport::new("backward", Duration::from_millis(400)));
    
        assert_eq!(report.children.len(), 2);
        assert_eq!(report.children[0].name, "dataloader");
    }
}
