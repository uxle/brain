//! # Checkpointing & Model State Tracking
//!
//! Save/restore model weights, track best validation scores, and manage checkpoint retention.

use std::collections::HashMap;
use brain_core::Tensor;

/// Metadata associated with a model checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointMeta {
    pub epoch: usize,
    pub step: usize,
    pub val_loss: f64,
    pub val_metric: f64,
    pub timestamp: u64,
}

/// Tracks the best model according to a validation metric or loss.
#[derive(Debug, Clone)]
pub struct BestModelTracker {
    pub best_score: f64,
    pub best_epoch: usize,
    pub minimize: bool,
    pub best_state: Option<HashMap<String, Tensor>>,
}

impl BestModelTracker {
    pub fn new(minimize: bool) -> Self {
        Self {
            best_score: if minimize { f64::INFINITY } else { f64::NEG_INFINITY },
            best_epoch: 0,
            minimize,
            best_state: None,
        }
    }

    /// Evaluates if `current_score` improves upon the historical best.
    pub fn check_improvement(&mut self, epoch: usize, current_score: f64, state: &HashMap<String, Tensor>) -> bool {
        let is_better = if self.minimize {
            current_score < self.best_score
        } else {
            current_score > self.best_score
        };

        if is_better {
            self.best_score = current_score;
            self.best_epoch = epoch;
            self.best_state = Some(state.clone());
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_best_model_tracker() {
        let mut tracker = BestModelTracker::new(true);
        let state1 = HashMap::new();
        assert!(tracker.check_improvement(1, 0.5, &state1));
        assert!(!tracker.check_improvement(2, 0.6, &state1));
        assert!(tracker.check_improvement(3, 0.3, &state1));
        assert_eq!(tracker.best_epoch, 3);
        assert_eq!(tracker.best_score, 0.3);
    }
}
