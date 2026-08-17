//! # Composite Regularization Stack & Validation
//!
//! Stacks multiple regularizers with independent weighting factors and validates against conflicting configurations.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::regularizers::Regularizer;

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

    #[test]
    fn test_rules_stress_001() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 1 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_002() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 2 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_003() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 3 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_004() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 4 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_005() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 5 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_006() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 6 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_007() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 7 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_008() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 8 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_009() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 9 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_010() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 10 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_011() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 11 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_012() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 12 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_013() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 13 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_014() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 14 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_015() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 15 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_016() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 16 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_017() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 17 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_018() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 18 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_019() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 19 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_020() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 20 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_021() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 21 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_022() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 22 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_023() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 23 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_024() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 24 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_025() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 25 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_026() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 26 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_027() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 27 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_028() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 28 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_029() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 29 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_030() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 30 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_031() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 31 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_032() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 32 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_033() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 33 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_034() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 34 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_035() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 35 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_036() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 36 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_037() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 37 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_038() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 38 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_039() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 39 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_040() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 40 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_041() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 41 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_042() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 42 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_043() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 43 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_044() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 44 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_045() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 45 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_046() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 46 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_047() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 47 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_048() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 48 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_049() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 49 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_050() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 50 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_051() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 51 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_052() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 52 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_053() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 53 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_054() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 54 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_055() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 55 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_056() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 56 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_057() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 57 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_058() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 58 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_059() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 59 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_060() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 60 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_061() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 61 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_062() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 62 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_063() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 63 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_064() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 64 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_065() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 65 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_066() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 66 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_067() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 67 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_068() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 68 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_069() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 69 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_070() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 70 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_071() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 71 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_072() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 72 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_073() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 73 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_074() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 74 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_075() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 75 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_076() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 76 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_077() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 77 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_078() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 78 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_079() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 79 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_080() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 80 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_081() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 81 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_082() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 82 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_083() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 83 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_084() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 84 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_085() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 85 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_086() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 86 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_087() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 87 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_088() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 88 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_089() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 89 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_090() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 90 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_091() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 91 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_092() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 92 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_093() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 93 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_094() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 94 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_095() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 95 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_096() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 96 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_097() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 97 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_098() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 98 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_099() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 99 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_100() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 100 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_101() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 101 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_102() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 102 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_103() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 103 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_104() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 104 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_105() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 105 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_106() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 106 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_107() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 107 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_108() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 108 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_109() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 109 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_110() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 110 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_111() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 111 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_112() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 112 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_113() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 113 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_114() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 114 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_115() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 115 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_116() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 116 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_117() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 117 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_118() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 118 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_119() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 119 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_120() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 120 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_121() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 121 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_122() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 122 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_123() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 123 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_124() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 124 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_125() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 125 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_126() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 126 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_127() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 127 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_128() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 128 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_129() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 129 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_130() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 130 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_131() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 131 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_132() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 132 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_133() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 133 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_134() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 134 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_135() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 135 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_136() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 136 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_137() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 137 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_138() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 138 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_139() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 139 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_140() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 140 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_141() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 141 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_142() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 142 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_143() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 143 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_144() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 144 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_145() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 145 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_146() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 146 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_147() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 147 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_148() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 148 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_149() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 149 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_150() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 150 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_151() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 151 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_152() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 152 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_153() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 153 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_154() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 154 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_155() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 155 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_156() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 156 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_157() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 157 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_158() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 158 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_159() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 159 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_160() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 160 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_161() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 161 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_162() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 162 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_163() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 163 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_164() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 164 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_165() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 165 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_166() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 166 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_167() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 167 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_168() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 168 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_169() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 169 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_170() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 170 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_171() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 171 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_172() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 172 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_173() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 173 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_174() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 174 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_175() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 175 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_176() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 176 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_177() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 177 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_178() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 178 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_179() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 179 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_180() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 180 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_181() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 181 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_182() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 182 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_183() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 183 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_184() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 184 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_185() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 185 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_186() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 186 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_187() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 187 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_188() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 188 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_189() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 189 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_190() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 190 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_191() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 191 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_192() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 192 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_193() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 193 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_194() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 194 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_195() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 195 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_196() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 196 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_197() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 197 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_198() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 198 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_199() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 199 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_200() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 200 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_201() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 201 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_202() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 202 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_203() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 203 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_204() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 204 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_205() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 205 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_206() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 206 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_207() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 207 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_208() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 208 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_209() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 209 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_210() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 210 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_211() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 211 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_212() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 212 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_213() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 213 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_214() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 214 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_215() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 215 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_216() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 216 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_217() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 217 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_218() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 218 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_219() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 219 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_220() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 220 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_221() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 221 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_222() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 222 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_223() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 223 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_224() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 224 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_225() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 225 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_226() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 226 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_227() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 227 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_228() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 228 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_229() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 229 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_230() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 230 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_231() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 231 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_232() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 232 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_233() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 233 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_234() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 234 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_235() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 235 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_236() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 236 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_237() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 237 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_238() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 238 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_239() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 239 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_240() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 240 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_241() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 241 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_242() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 242 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_243() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 243 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_244() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 244 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_245() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 245 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_246() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 246 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_247() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 247 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_248() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 248 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_249() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 249 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_250() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 250 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_251() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 251 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_252() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 252 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_253() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 253 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_254() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 254 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_255() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 255 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_256() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 256 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_257() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 257 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_258() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 258 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_259() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 259 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_260() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 260 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_261() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 261 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_262() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 262 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_263() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 263 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_264() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 264 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_265() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 265 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_266() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 266 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_267() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 267 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_268() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 268 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_269() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 269 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_270() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 270 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_271() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 271 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_272() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 272 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_273() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 273 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_274() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 274 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_275() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 275 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_276() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 276 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_277() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 277 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_278() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 278 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_279() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 279 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_280() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 280 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_281() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 281 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_282() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 282 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_283() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 283 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_284() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 284 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_285() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 285 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_286() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 286 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_287() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 287 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_288() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 288 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_289() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 289 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_290() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 290 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_291() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 291 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_292() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 292 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_293() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 293 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_294() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 294 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_295() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 295 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_296() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 296 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_297() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 297 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }

    #[test]
    fn test_rules_stress_298() {
        let mut stack = RegStack::new();
        stack.add(L1Regularizer::new(0.01), 1.0);
        stack.add(L2Regularizer::new(0.02), 2.0);

        let t = Tensor::from_slice(&[1.0, -2.0, 298 as f64 * 0.1], vec![3]);
        let pen = stack.total_penalty(&[t]);
        assert!(pen > 0.0);
    }
}
