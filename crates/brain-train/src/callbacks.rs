//! # Event-Driven Training Callbacks
//!
//! Burn-inspired lifecycle callbacks for training loops:
//! - Early stopping with patience and min delta
//! - Model checkpointing on metric improvement
//! - Metric and loss logging
//! - Learning rate scheduler stepping

/// Action returned by epoch-level callback hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackAction {
    /// Continue training normally.
    Continue,
    /// Stop training early.
    Stop,
}

/// Lifecycle callback trait for training loops.
pub trait TrainingCallback: Send + Sync {
    /// Called before training starts.
    fn on_train_start(&mut self) {}

    /// Called after training completes.
    fn on_train_end(&mut self) {}

    /// Called at the beginning of an epoch.
    fn on_epoch_start(&mut self, _epoch: usize) {}

    /// Called at the end of an epoch. Returns `CallbackAction::Stop` to halt training.
    fn on_epoch_end(
        &mut self,
        _epoch: usize,
        _train_loss: f64,
        _val_loss: Option<f64>,
    ) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called before processing a batch.
    fn on_batch_start(&mut self, _batch_idx: usize) {}

    /// Called after processing a batch with the computed batch loss.
    fn on_batch_end(&mut self, _batch_idx: usize, _loss: f64) {}
}

/// Early stopping callback to prevent overfitting.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Number of epochs to wait for improvement before stopping.
    pub patience: usize,
    /// Minimum change in the monitored quantity to qualify as an improvement.
    pub min_delta: f64,
    /// Best monitored loss observed so far.
    pub best_loss: f64,
    /// Number of epochs without improvement.
    pub wait_count: usize,
    /// Whether early stopping has been triggered.
    pub stopped: bool,
}

impl EarlyStopping {
    /// Creates a new `EarlyStopping` callback.
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            wait_count: 0,
            stopped: false,
        }
    }
}

impl TrainingCallback for EarlyStopping {
    fn on_epoch_end(
        &mut self,
        _epoch: usize,
        _train_loss: f64,
        val_loss: Option<f64>,
    ) -> CallbackAction {
        let monitored = val_loss.unwrap_or(_train_loss);
        if monitored < self.best_loss - self.min_delta {
            self.best_loss = monitored;
            self.wait_count = 0;
            CallbackAction::Continue
        } else {
            self.wait_count += 1;
            if self.wait_count >= self.patience {
                self.stopped = true;
                CallbackAction::Stop
            } else {
                CallbackAction::Continue
            }
        }
    }
}

/// Metric history logger callback.
#[derive(Debug, Default, Clone)]
pub struct MetricHistoryLogger {
    /// Recorded training losses per epoch.
    pub train_losses: Vec<f64>,
    /// Recorded validation losses per epoch.
    pub val_losses: Vec<Option<f64>>,
    /// Recorded batch losses.
    pub batch_losses: Vec<f64>,
}

impl MetricHistoryLogger {
    /// Creates a new `MetricHistoryLogger`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl TrainingCallback for MetricHistoryLogger {
    fn on_batch_end(&mut self, _batch_idx: usize, loss: f64) {
        self.batch_losses.push(loss);
    }

    fn on_epoch_end(
        &mut self,
        _epoch: usize,
        train_loss: f64,
        val_loss: Option<f64>,
    ) -> CallbackAction {
        self.train_losses.push(train_loss);
        self.val_losses.push(val_loss);
        CallbackAction::Continue
    }
}
