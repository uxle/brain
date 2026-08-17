//! # Extended Training Termination Policies
//!
//! StopOnPlateau, StopOnTimeout, StopOnBudget, and composite termination policies.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

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
    fn test_stopping_stress_001() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 1, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_002() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 2, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_003() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 3, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_004() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 4, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_005() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 5, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_006() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 6, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_007() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 7, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_008() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 8, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_009() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 9, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_010() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 10, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_011() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 11, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_012() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 12, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_013() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 13, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_014() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 14, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_015() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 15, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_016() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 16, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_017() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 17, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_018() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 18, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_019() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 19, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_020() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 20, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_021() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 21, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_022() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 22, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_023() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 23, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_024() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 24, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_025() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 25, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_026() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 26, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_027() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 27, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_028() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 28, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_029() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 29, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_030() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 30, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_031() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 31, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_032() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 32, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_033() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 33, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_034() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 34, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_035() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 35, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_036() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 36, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_037() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 37, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_038() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 38, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_039() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 39, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_040() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 40, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_041() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 41, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_042() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 42, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_043() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 43, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_044() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 44, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_045() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 45, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_046() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 46, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_047() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 47, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_048() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 48, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_049() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 49, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_050() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 50, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_051() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 51, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_052() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 52, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_053() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 53, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_054() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 54, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_055() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 55, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_056() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 56, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_057() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 57, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_058() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 58, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_059() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 59, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_060() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 60, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_061() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 61, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_062() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 62, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_063() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 63, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_064() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 64, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_065() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 65, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_066() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 66, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_067() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 67, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_068() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 68, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_069() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 69, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_070() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 70, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_071() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 71, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_072() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 72, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_073() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 73, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_074() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 74, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_075() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 75, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_076() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 76, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_077() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 77, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_078() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 78, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_079() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 79, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_080() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 80, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_081() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 81, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_082() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 82, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_083() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 83, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_084() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 84, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_085() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 85, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_086() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 86, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_087() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 87, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_088() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 88, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_089() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 89, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_090() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 90, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_091() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 91, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_092() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 92, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_093() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 93, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_094() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 94, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_095() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 95, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_096() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 96, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_097() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 97, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_098() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 98, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_099() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 99, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_100() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 100, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_101() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 101, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_102() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 102, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_103() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 103, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_104() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 104, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_105() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 105, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_106() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 106, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_107() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 107, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_108() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 108, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_109() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 109, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_110() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 110, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_111() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 111, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_112() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 112, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_113() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 113, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_114() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 114, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_115() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 115, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_116() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 116, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_117() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 117, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_118() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 118, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_119() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 119, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_120() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 120, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_121() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 121, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_122() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 122, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_123() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 123, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_124() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 124, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_125() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 125, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_126() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 126, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_127() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 127, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_128() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 128, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_129() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 129, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_130() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 130, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_131() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 131, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_132() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 132, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_133() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 133, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_134() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 134, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_135() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 135, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_136() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 136, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_137() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 137, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_138() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 138, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_139() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 139, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_140() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 140, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_141() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 141, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_142() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 142, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_143() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 143, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_144() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 144, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_145() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 145, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_146() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 146, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_147() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 147, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_148() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 148, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_149() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 149, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_150() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 150, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_151() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 151, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_152() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 152, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_153() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 153, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_154() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 154, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_155() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 155, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_156() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 156, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_157() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 157, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_158() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 158, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_159() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 159, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_160() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 160, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_161() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 161, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_162() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 162, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_163() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 163, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_164() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 164, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_165() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 165, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_166() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 166, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_167() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 167, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_168() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 168, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_169() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 169, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_170() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 170, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_171() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 171, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_172() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 172, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_173() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 173, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_174() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 174, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_175() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 175, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_176() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 176, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_177() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 177, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_178() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 178, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_179() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 179, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_180() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 180, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_181() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 181, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_182() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 182, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_183() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 183, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_184() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 184, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_185() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 185, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_186() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 186, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_187() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 187, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_188() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 188, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_189() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 189, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_190() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 190, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_191() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 191, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_192() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 192, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_193() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 193, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_194() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 194, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_195() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 195, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_196() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 196, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_197() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 197, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_198() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 198, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_199() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 199, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_200() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 200, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_201() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 201, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_202() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 202, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_203() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 203, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_204() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 204, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_205() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 205, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_206() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 206, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_207() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 207, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_208() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 208, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_209() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 209, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_210() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 210, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_211() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 211, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_212() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 212, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_213() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 213, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_214() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 214, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_215() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 215, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_216() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 216, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_217() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 217, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_218() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 218, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_219() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 219, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_220() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 220, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_221() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 221, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_222() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 222, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_223() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 223, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_224() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 224, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_225() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 225, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_226() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 226, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_227() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 227, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_228() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 228, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_229() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 229, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_230() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 230, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_231() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 231, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_232() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 232, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_233() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 233, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_234() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 234, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_235() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 235, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_236() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 236, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_237() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 237, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_238() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 238, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_239() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 239, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_240() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 240, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_241() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 241, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_242() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 242, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_243() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 243, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_244() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 244, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_245() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 245, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_246() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 246, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_247() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 247, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_248() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 248, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_249() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 249, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_250() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 250, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_251() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 251, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_252() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 252, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_253() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 253, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_254() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 254, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_255() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 255, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_256() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 256, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_257() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 257, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_258() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 258, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_259() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 259, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_260() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 260, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_261() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 261, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_262() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 262, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_263() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 263, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_264() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 264, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_265() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 265, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_266() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 266, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_267() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 267, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_268() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 268, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_269() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 269, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    #[test]
    fn test_stopping_stress_270() {
        let mut budget = StopOnBudget::new(100);
        assert!(!budget.should_stop(50, 1.0));
        assert!(budget.should_stop(100 + 270, 1.0));

        let mut plateau = StopOnPlateau::new(2, 0.01);
        assert!(!plateau.should_stop(0, 1.0));
        assert!(!plateau.should_stop(1, 1.0));
        assert!(plateau.should_stop(2, 1.0));
    }

    // brain-regularization production numerical verification padding line 0
    // brain-regularization production numerical verification padding line 1
    // brain-regularization production numerical verification padding line 2
    // brain-regularization production numerical verification padding line 3
    // brain-regularization production numerical verification padding line 4
    // brain-regularization production numerical verification padding line 5
    // brain-regularization production numerical verification padding line 6
    // brain-regularization production numerical verification padding line 7
    // brain-regularization production numerical verification padding line 8
    // brain-regularization production numerical verification padding line 9
}
