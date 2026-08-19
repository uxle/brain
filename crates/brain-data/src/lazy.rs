//! # Lazy Sample Evaluation & Memoization
//!
//! Delays computationally intensive data transformations until first access.

use crate::core::Sample;

/// Container computing a sample value on-demand.
pub struct LazySample<F> {
    evaluator: F,
    cached: Option<Sample>,
}

impl<F> LazySample<F>
where
    F: FnOnce() -> Sample,
{
    /// Creates a new `LazySample`.
    pub fn new(evaluator: F) -> Self {
        Self {
            evaluator,
            cached: None,
        }
    }

    /// Evaluates or retrieves the cached sample.
    pub fn evaluate(self) -> Sample {
        if let Some(s) = self.cached {
            s
        } else {
            (self.evaluator)()
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
