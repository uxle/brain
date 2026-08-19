//! # Metric Tracker & Accumulator
//!
//! Live epoch accumulator with lifecycle management (reset, update, epoch summary).
#![allow(missing_docs)]

use std::collections::HashMap;

/// Live incremental metric accumulator for training/eval loops.
#[derive(Debug, Default)]
pub struct MetricTracker {
    sums: HashMap<String, f64>,
    counts: HashMap<String, usize>,
}

impl MetricTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, name: &str, value: f64, count: usize) {
        *self.sums.entry(name.to_string()).or_insert(0.0) += value * count as f64;
        *self.counts.entry(name.to_string()).or_insert(0) += count;
    }

    pub fn mean(&self, name: &str) -> Option<f64> {
        let sum = self.sums.get(name)?;
        let count = self.counts.get(name)?;
        if *count > 0 { Some(sum / *count as f64) } else { None }
    }

    pub fn summary(&self) -> HashMap<String, f64> {
        let mut res = HashMap::new();
        for (k, &sum) in &self.sums {
            if let Some(&cnt) = self.counts.get(k) {
                if cnt > 0 {
                    res.insert(k.clone(), sum / cnt as f64);
                }
            }
        }
        res
    }

    pub fn reset(&mut self) {
        self.sums.clear();
        self.counts.clear();
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
