//! # Early Stopping Mechanism
//!
//! Monitored metric tracking with patience counters, delta thresholds, and best-state restoration.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;

/// Optimization direction mode for early stopping metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetricMode {
    #[default]
    Min,
    Max,
}

/// Configuration settings for EarlyStopping monitor.
#[derive(Debug, Clone, PartialEq)]
pub struct EarlyStopConfig {
    pub patience: usize,
    pub min_delta: f64,
    pub mode: MetricMode,
    pub restore_best_weights: bool,
}

impl Default for EarlyStopConfig {
    fn default() -> Self {
        Self {
            patience: 5,
            min_delta: 1e-4,
            mode: MetricMode::Min,
            restore_best_weights: true,
        }
    }
}

/// State tracker for EarlyStopping monitor.
#[derive(Debug, Clone, PartialEq)]
pub struct EarlyStopState {
    pub best_metric: f64,
    pub best_epoch: usize,
    pub wait_count: usize,
    pub stopped: bool,
}

/// Early Stopping Engine.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    pub config: EarlyStopConfig,
    pub state: EarlyStopState,
    pub best_weights: Option<Vec<Tensor>>,
}

impl EarlyStopping {
    pub fn new(config: EarlyStopConfig) -> Self {
        let best_metric = match config.mode {
            MetricMode::Min => f64::INFINITY,
            MetricMode::Max => f64::NEG_INFINITY,
        };

        Self {
            config,
            state: EarlyStopState {
                best_metric,
                best_epoch: 0,
                wait_count: 0,
                stopped: false,
            },
            best_weights: None,
        }
    }

    /// Feeds epoch evaluation metric and optional checkpoint weights to check stopping criteria.
    pub fn step(&mut self, epoch: usize, metric: f64, weights: Option<&[Tensor]>) -> bool {
        if self.state.stopped {
            return true;
        }

        let is_better = match self.config.mode {
            MetricMode::Min => metric < self.state.best_metric - self.config.min_delta,
            MetricMode::Max => metric > self.state.best_metric + self.config.min_delta,
        };

        if is_better {
            self.state.best_metric = metric;
            self.state.best_epoch = epoch;
            self.state.wait_count = 0;
            if self.config.restore_best_weights {
                self.best_weights = weights.map(|w| w.to_vec());
            }
        } else {
            self.state.wait_count += 1;
            if self.state.wait_count >= self.config.patience {
                self.state.stopped = true;
            }
        }

        self.state.stopped
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::dropout::*;
    use crate::normalization::*;
    use crate::regularizers::*;
    use crate::decay::*;
    use crate::earlystop::*;
    use crate::stopping::*;
    use crate::augment::*;
    use crate::perturb::*;
    use crate::dropout_uncertainty::*;
    use crate::label_smooth::*;
    use crate::curriculum::*;
    use crate::consistency::*;
    use crate::rules::*;
    use crate::registry::*;
    use crate::train_hooks::*;
    use crate::ops::*;
    use crate::r#impl::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
