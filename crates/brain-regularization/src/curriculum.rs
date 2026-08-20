//! # Curriculum Regularization
//!
//! Progressively anneals regularization strength (e.g. ramping dropout probability p(t) or weight decay).
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

/// Configuration for curriculum schedules.
#[derive(Debug, Clone, PartialEq)]
pub struct CurriculumConfig {
    pub initial_value: f64,
    pub final_value: f64,
    pub total_steps: usize,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            initial_value: 0.0,
            final_value: 0.5,
            total_steps: 1000,
        }
    }
}

/// Curriculum scheduler gradually scaling regularization strength.
#[derive(Debug, Clone)]
pub struct CurriculumScheduler {
    pub config: CurriculumConfig,
}

impl CurriculumScheduler {
    pub fn new(config: CurriculumConfig) -> Self {
        Self { config }
    }

    /// Computes annealed regularization parameter at training step t.
    pub fn get_value(&self, step: usize) -> f64 {
        if step >= self.config.total_steps {
            return self.config.final_value;
        }
        let progress = step as f64 / self.config.total_steps.max(1) as f64;
        self.config.initial_value + (self.config.final_value - self.config.initial_value) * progress
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
