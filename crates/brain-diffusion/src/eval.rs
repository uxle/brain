//! # Generative Quality Evaluation
//!
//! Feature distance metrics and per-step quality curves.

/// Evaluation metrics report.
#[derive(Debug, Clone, Default)]
pub struct EvalReport {
    pub step_count: usize,
}

impl EvalReport {
    /// Creates a new `EvalReport`.
    pub fn new(step_count: usize) -> Self {
        Self { step_count }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
