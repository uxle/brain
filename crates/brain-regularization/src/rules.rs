//! # Composite Regularization Stack & Validation
//!
//! Stacks multiple regularizers with independent weighting factors and validates against conflicting configurations.
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

use super::regularizers::Regularizer;
use brain_core::Tensor;

/// An item in the regularization stack holding a regularizer and its loss multiplier weight.
pub struct WeightedRegularizer {
    pub regularizer: Box<dyn Regularizer>,
    pub weight: f64,
}

/// Stack of explicit parameter penalty regularizers.
#[derive(Default)]
pub struct RegStack {
    pub items: Vec<WeightedRegularizer>,
}

impl RegStack {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Appends a regularizer with scaling weight to the stack.
    pub fn add<R: Regularizer + 'static>(&mut self, regularizer: R, weight: f64) {
        self.items.push(WeightedRegularizer {
            regularizer: Box::new(regularizer),
            weight: weight.max(0.0),
        });
    }

    /// Computes aggregated total penalty scalar across all stacked regularizers.
    pub fn total_penalty(&self, params: &[Tensor]) -> f64 {
        let mut total = 0.0;
        for item in &self.items {
            total += item.weight * item.regularizer.penalty(params);
        }
        total
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
