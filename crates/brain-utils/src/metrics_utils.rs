//! # Lightweight Metrics Registry
//!
//! Provides metric registries, typed metrics (Counter, Gauge, Histogram),
//! labeled metric keys, and Prometheus-compatible text serialization.

use std::collections::BTreeMap;
use std::sync::RwLock;

/// Supported metric types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Monotonically increasing value.
    Counter,
    /// Variable instantaneous value.
    Gauge,
    /// Statistical distribution buckets.
    Histogram,
}

/// A registered metric item with description and labels.
#[derive(Debug, Clone, PartialEq)]
pub struct Metric {
    /// Metric name (e.g. `loss_epoch_total`).
    pub name: String,
    /// Metric type.
    pub metric_type: MetricType,
    /// Human-readable help docstring.
    pub help: String,
    /// Key-value dimension labels.
    pub labels: BTreeMap<String, String>,
    /// Instantaneous float value.
    pub value: f64,
}

impl Metric {
    /// Creates a new metric item.
    pub fn new(name: &str, metric_type: MetricType, help: &str) -> Self {
        Self {
            name: name.to_string(),
            metric_type,
            help: help.to_string(),
            labels: BTreeMap::new(),
            value: 0.0,
        }
    }

    /// Attaches a label.
    pub fn with_label(mut self, key: &str, val: &str) -> Self {
        self.labels.insert(key.to_string(), val.to_string());
        self
    }
}

/// Central registry for managing collection of application metrics.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    metrics: RwLock<BTreeMap<String, Metric>>,
}

impl MetricsRegistry {
    /// Constructs a new empty metrics registry.
    pub fn new() -> Self {
        Self {
            metrics: RwLock::new(BTreeMap::new()),
        }
    }

    /// Registers or updates a metric value.
    pub fn set_metric(&self, mut metric: Metric, value: f64) {
        metric.value = value;
        let mut w = self.metrics.write().unwrap();
        w.insert(metric.name.clone(), metric);
    }

    /// Increments a counter metric.
    pub fn inc_counter(&self, name: &str, delta: f64) {
        let mut w = self.metrics.write().unwrap();
        if let Some(m) = w.get_mut(name) {
            m.value += delta;
        } else {
            let mut m = Metric::new(name, MetricType::Counter, "");
            m.value = delta;
            w.insert(name.to_string(), m);
        }
    }

    /// Sets a gauge metric value.
    pub fn set_gauge(&self, name: &str, value: f64) {
        let mut w = self.metrics.write().unwrap();
        if let Some(m) = w.get_mut(name) {
            m.value = value;
        } else {
            let mut m = Metric::new(name, MetricType::Gauge, "");
            m.value = value;
            w.insert(name.to_string(), m);
        }
    }

    /// Retrieves metric value.
    pub fn get_value(&self, name: &str) -> Option<f64> {
        self.metrics.read().unwrap().get(name).map(|m| m.value)
    }

    /// Exports all metrics as a Prometheus text format report.
    pub fn export_prometheus_text(&self) -> String {
        let mut out = String::new();
        let r = self.metrics.read().unwrap();
        for (name, m) in r.iter() {
            let type_str = match m.metric_type {
                MetricType::Counter => "counter",
                MetricType::Gauge => "gauge",
                MetricType::Histogram => "histogram",
            };
            if !m.help.is_empty() {
                out.push_str(&format!("# HELP {} {}\n", name, m.help));
            }
            out.push_str(&format!("# TYPE {} {}\n", name, type_str));
            if m.labels.is_empty() {
                out.push_str(&format!("{} {}\n", name, m.value));
            } else {
                let label_str: Vec<String> = m
                    .labels
                    .iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                    .collect();
                out.push_str(&format!(
                    "{}{{{}}} {}\n",
                    name,
                    label_str.join(","),
                    m.value
                ));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_metrics_registry_and_export_1() {
        let registry = MetricsRegistry::new();
        registry.inc_counter("tokens_processed", 100.0);
        registry.inc_counter("tokens_processed", 50.0);
        assert_eq!(registry.get_value("tokens_processed"), Some(150.0));

        registry.set_gauge("memory_usage_mb", 2048.5);
        assert_eq!(registry.get_value("memory_usage_mb"), Some(2048.5));

        let prom = registry.export_prometheus_text();
        assert!(prom.contains("tokens_processed"));
        assert!(prom.contains("memory_usage_mb"));
    }
}
