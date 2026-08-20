//! # Extended Training Termination Policies
//!
//! StopOnPlateau, StopOnTimeout, StopOnBudget, and composite termination policies.
#![allow(
    missing_docs,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

/// Action to execute upon termination condition trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StopAction {
    #[default]
    HaltTraining,
    ReduceLearningRate,
    WarnOnly,
}

/// Policy for training termination.
pub trait StopPolicy: Send + Sync {
    /// Checks if training should be halted.
    fn should_stop(&mut self, step: usize, loss: f64) -> bool;
}

/// Halts training when step or FLOP budget is exhausted.
#[derive(Debug, Clone)]
pub struct StopOnBudget {
    pub max_steps: usize,
}

impl StopOnBudget {
    pub fn new(max_steps: usize) -> Self {
        Self { max_steps }
    }
}

impl StopPolicy for StopOnBudget {
    fn should_stop(&mut self, step: usize, _loss: f64) -> bool {
        step >= self.max_steps
    }
}

/// Halts training when metric has plateaued beyond tolerance.
#[derive(Debug, Clone)]
pub struct StopOnPlateau {
    pub patience: usize,
    pub min_loss_delta: f64,
    pub best_loss: f64,
    pub plateau_count: usize,
}

impl StopOnPlateau {
    pub fn new(patience: usize, min_loss_delta: f64) -> Self {
        Self {
            patience,
            min_loss_delta,
            best_loss: f64::INFINITY,
            plateau_count: 0,
        }
    }
}

impl StopPolicy for StopOnPlateau {
    fn should_stop(&mut self, _step: usize, loss: f64) -> bool {
        if loss < self.best_loss - self.min_loss_delta {
            self.best_loss = loss;
            self.plateau_count = 0;
            false
        } else {
            self.plateau_count += 1;
            self.plateau_count >= self.patience
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown
    )]
    use super::*;
    use crate::augment::*;
    use crate::config::*;
    use crate::consistency::*;
    use crate::core::*;
    use crate::curriculum::*;
    use crate::decay::*;
    use crate::dropout::*;
    use crate::dropout_uncertainty::*;
    use crate::earlystop::*;
    use crate::label_smooth::*;
    use crate::normalization::*;
    use crate::ops::*;
    use crate::perturb::*;
    use crate::r#impl::*;
    use crate::registry::*;
    use crate::regularizers::*;
    use crate::rules::*;
    use crate::stopping::*;
    use crate::train_hooks::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
