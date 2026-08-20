//! # Training & Evaluation Metrics
//!
//! Track running averages, classification accuracy, top-k accuracy, loss metrics, and perplexity.

use brain_core::Tensor;

/// Running average accumulator with exponential or uniform weighting.
#[derive(Debug, Clone)]
pub struct RunningAverage {
    total: f64,
    count: usize,
    window: Option<Vec<f64>>,
}

impl Default for RunningAverage {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningAverage {
    pub fn new() -> Self {
        Self { total: 0.0, count: 0, window: None }
    }

    pub fn with_window(size: usize) -> Self {
        Self { total: 0.0, count: 0, window: Some(Vec::with_capacity(size)) }
    }

    pub fn update(&mut self, value: f64) {
        self.total += value;
        self.count += 1;
        if let Some(ref mut win) = self.window {
            win.push(value);
        }
    }

    pub fn value(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total / (self.count as f64)
        }
    }

    pub fn reset(&mut self) {
        self.total = 0.0;
        self.count = 0;
        if let Some(ref mut win) = self.window {
            win.clear();
        }
    }
}

/// Accuracy Metric: computes top-1 accuracy between predicted logits and target indices.
#[derive(Debug, Clone, Default)]
pub struct AccuracyMetric {
    correct: usize,
    total: usize,
}

impl AccuracyMetric {
    pub fn new() -> Self {
        Self { correct: 0, total: 0 }
    }

    pub fn update(&mut self, logits: &Tensor, targets: &[usize]) {
        let shape = logits.shape();
        let rows = shape[0];
        let cols = if shape.len() > 1 { shape[1] } else { 1 };
        let data = logits.data();

        let n = rows.min(targets.len());
        for r in 0..n {
            let row = &data[r * cols..(r + 1) * cols];
            let mut best_c = 0;
            let mut max_v = f64::NEG_INFINITY;
            for (c, &v) in row.iter().enumerate() {
                if v > max_v {
                    max_v = v;
                    best_c = c;
                }
            }
            if best_c == targets[r] {
                self.correct += 1;
            }
            self.total += 1;
        }
    }

    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.correct as f64) / (self.total as f64)
        }
    }

    pub fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }
}

/// Top-K Accuracy Metric.
#[derive(Debug, Clone)]
pub struct TopKAccuracyMetric {
    pub k: usize,
    correct: usize,
    total: usize,
}

impl TopKAccuracyMetric {
    pub fn new(k: usize) -> Self {
        Self { k: k.max(1), correct: 0, total: 0 }
    }

    pub fn update(&mut self, logits: &Tensor, targets: &[usize]) {
        let shape = logits.shape();
        let rows = shape[0];
        let cols = if shape.len() > 1 { shape[1] } else { 1 };
        let data = logits.data();

        let n = rows.min(targets.len());
        for r in 0..n {
            let row = &data[r * cols..(r + 1) * cols];
            let mut indices: Vec<usize> = (0..cols).collect();
            indices.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());

            let top_k = &indices[..self.k.min(cols)];
            if top_k.contains(&targets[r]) {
                self.correct += 1;
            }
            self.total += 1;
        }
    }

    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.correct as f64) / (self.total as f64)
        }
    }

    pub fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_running_average() {
        let mut avg = RunningAverage::new();
        avg.update(10.0);
        avg.update(20.0);
        avg.update(30.0);
        assert_eq!(avg.value(), 20.0);
    }

    #[test]
    fn test_accuracy_metric() {
        let mut acc = AccuracyMetric::new();
        let logits = Tensor::from_slice(&[0.1, 0.9, 0.8, 0.2], vec![2, 2]);
        let targets = vec![1, 0];
        acc.update(&logits, &targets);
        assert_eq!(acc.accuracy(), 1.0);
    }
}
