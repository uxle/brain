//! # Metric Counters and Gauges
//!
//! Provides thread-safe atomic counters, gauges, and metric snapshot dictionaries.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Monotonically increasing 64-bit atomic counter.
#[derive(Debug, Default)]
pub struct AtomicCounter {
    val: AtomicU64,
}

impl AtomicCounter {
    /// Creates a new counter starting at 0.
    pub fn new() -> Self {
        Self {
            val: AtomicU64::new(0),
        }
    }

    /// Increments by 1.
    pub fn inc(&self) -> u64 {
        self.val.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Adds delta to counter.
    pub fn add(&self, delta: u64) -> u64 {
        self.val.fetch_add(delta, Ordering::Relaxed) + delta
    }

    /// Retrieves current value.
    pub fn get(&self) -> u64 {
        self.val.load(Ordering::Relaxed)
    }

    /// Resets counter to 0.
    pub fn reset(&self) {
        self.val.store(0, Ordering::Relaxed);
    }
}

/// Value-variable 64-bit atomic gauge.
#[derive(Debug, Default)]
pub struct AtomicGauge {
    val: AtomicI64,
}

impl AtomicGauge {
    /// Creates a new gauge starting at 0.
    pub fn new() -> Self {
        Self {
            val: AtomicI64::new(0),
        }
    }

    /// Sets the gauge to a specific value.
    pub fn set(&self, val: i64) {
        self.val.store(val, Ordering::Relaxed);
    }

    /// Increments by 1.
    pub fn inc(&self) -> i64 {
        self.val.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrements by 1.
    pub fn dec(&self) -> i64 {
        self.val.fetch_sub(1, Ordering::Relaxed) - 1
    }

    /// Retrieves current value.
    pub fn get(&self) -> i64 {
        self.val.load(Ordering::Relaxed)
    }
}

/// Registry set of named counters and gauges.
#[derive(Debug, Default)]
pub struct CounterSet {
    counters: RwLock<BTreeMap<String, Arc<AtomicCounter>>>,
    gauges: RwLock<BTreeMap<String, Arc<AtomicGauge>>>,
}

impl CounterSet {
    /// Creates an empty counter set.
    pub fn new() -> Self {
        Self {
            counters: RwLock::new(BTreeMap::new()),
            gauges: RwLock::new(BTreeMap::new()),
        }
    }

    /// Gets or creates a named atomic counter.
    pub fn get_counter(&self, name: &str) -> Arc<AtomicCounter> {
        let mut w = self.counters.write().unwrap();
        w.entry(name.to_string())
            .or_insert_with(|| Arc::new(AtomicCounter::new()))
            .clone()
    }

    /// Gets or creates a named atomic gauge.
    pub fn get_gauge(&self, name: &str) -> Arc<AtomicGauge> {
        let mut w = self.gauges.write().unwrap();
        w.entry(name.to_string())
            .or_insert_with(|| Arc::new(AtomicGauge::new()))
            .clone()
    }

    /// Takes a snapshot of all counter values.
    pub fn snapshot_counters(&self) -> BTreeMap<String, u64> {
        let r = self.counters.read().unwrap();
        r.iter().map(|(k, v)| (k.clone(), v.get())).collect()
    }

    /// Takes a snapshot of all gauge values.
    pub fn snapshot_gauges(&self) -> BTreeMap<String, i64> {
        let r = self.gauges.read().unwrap();
        r.iter().map(|(k, v)| (k.clone(), v.get())).collect()
    }

    /// Resets all counters and gauges.
    pub fn reset_all(&self) {
        if let Ok(r) = self.counters.read() {
            for v in r.values() {
                v.reset();
            }
        }
        if let Ok(r) = self.gauges.read() {
            for v in r.values() {
                v.set(0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_counters_and_gauges_1() {
        let set = CounterSet::new();
        let c = set.get_counter("http_requests");
        c.inc();
        c.add(9);
        assert_eq!(c.get(), 10);

        let g = set.get_gauge("active_connections");
        g.set(5);
        g.inc();
        g.dec();
        assert_eq!(g.get(), 5);

        let c_snap = set.snapshot_counters();
        assert_eq!(c_snap.get("http_requests"), Some(&10));

        let g_snap = set.snapshot_gauges();
        assert_eq!(g_snap.get("active_connections"), Some(&5));

        set.reset_all();
        assert_eq!(c.get(), 0);
        assert_eq!(g.get(), 0);
    }
}
