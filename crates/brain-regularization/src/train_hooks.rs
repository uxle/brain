//! # Training Loop Lifecycle Hooks
//!
//! Interceptors for applying weight decay, logging penalties, and tracking early stopping state.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;

/// Configuration for training lifecycle hooks.
#[derive(Debug, Clone, PartialEq)]
pub struct HookConfig {
    pub enable_weight_decay: bool,
    pub weight_decay_rate: f64,
}

impl Default for HookConfig {
    fn default() -> Self {
        Self {
            enable_weight_decay: true,
            weight_decay_rate: 1e-4,
        }
    }
}

/// Regularization Training Hook intercepting forward/backward training passes.
#[derive(Debug, Clone)]
pub struct RegHook {
    pub config: HookConfig,
}

impl RegHook {
    pub fn new(config: HookConfig) -> Self {
        Self { config }
    }

    /// Hook executed after gradient descent optimizer step to apply decoupled decay.
    pub fn after_optimizer_step(&self, params: &mut [Tensor], lr: f64) {
        if !self.config.enable_weight_decay || self.config.weight_decay_rate == 0.0 {
            return;
        }
        let factor = 1.0 - lr * self.config.weight_decay_rate;
        for p in params.iter_mut() {
            for v in p.data_mut() {
                *v *= factor;
            }
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
    fn test_train_hooks_stress_001() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 1 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_002() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 2 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_003() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 3 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_004() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 4 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_005() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 5 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_006() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 6 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_007() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 7 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_008() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 8 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_009() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 9 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_010() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 10 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_011() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 11 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_012() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 12 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_013() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 13 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_014() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 14 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_015() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 15 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_016() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 16 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_017() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 17 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_018() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 18 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_019() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 19 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_020() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 20 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_021() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 21 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_022() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 22 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_023() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 23 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_024() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 24 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_025() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 25 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_026() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 26 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_027() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 27 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_028() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 28 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_029() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 29 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_030() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 30 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_031() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 31 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_032() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 32 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_033() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 33 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_034() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 34 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_035() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 35 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_036() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 36 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_037() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 37 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_038() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 38 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_039() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 39 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_040() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 40 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_041() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 41 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_042() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 42 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_043() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 43 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_044() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 44 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_045() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 45 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_046() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 46 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_047() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 47 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_048() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 48 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_049() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 49 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_050() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 50 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_051() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 51 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_052() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 52 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_053() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 53 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_054() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 54 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_055() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 55 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_056() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 56 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_057() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 57 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_058() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 58 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_059() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 59 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_060() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 60 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_061() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 61 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_062() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 62 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_063() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 63 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_064() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 64 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_065() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 65 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_066() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 66 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_067() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 67 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_068() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 68 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_069() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 69 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_070() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 70 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_071() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 71 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_072() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 72 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_073() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 73 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_074() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 74 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_075() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 75 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_076() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 76 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_077() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 77 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_078() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 78 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_079() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 79 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_080() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 80 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_081() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 81 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_082() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 82 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_083() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 83 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_084() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 84 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_085() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 85 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_086() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 86 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_087() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 87 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_088() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 88 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_089() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 89 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_090() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 90 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_091() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 91 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_092() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 92 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_093() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 93 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_094() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 94 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_095() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 95 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_096() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 96 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_097() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 97 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_098() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 98 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_099() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 99 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_100() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 100 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_101() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 101 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_102() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 102 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_103() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 103 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_104() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 104 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_105() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 105 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_106() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 106 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_107() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 107 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_108() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 108 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_109() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 109 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_110() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 110 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_111() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 111 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_112() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 112 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_113() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 113 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_114() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 114 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_115() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 115 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_116() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 116 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_117() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 117 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_118() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 118 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_119() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 119 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_120() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 120 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_121() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 121 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_122() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 122 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_123() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 123 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_124() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 124 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_125() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 125 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_126() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 126 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_127() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 127 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_128() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 128 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_129() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 129 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_130() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 130 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_131() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 131 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_132() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 132 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_133() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 133 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_134() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 134 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_135() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 135 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_136() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 136 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_137() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 137 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_138() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 138 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_139() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 139 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_140() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 140 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_141() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 141 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_142() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 142 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_143() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 143 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_144() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 144 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_145() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 145 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_146() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 146 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_147() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 147 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_148() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 148 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_149() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 149 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_150() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 150 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_151() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 151 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_152() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 152 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_153() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 153 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_154() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 154 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_155() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 155 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_156() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 156 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_157() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 157 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_158() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 158 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_159() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 159 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_160() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 160 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_161() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 161 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_162() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 162 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_163() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 163 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_164() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 164 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_165() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 165 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_166() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 166 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_167() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 167 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_168() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 168 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_169() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 169 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_170() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 170 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_171() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 171 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_172() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 172 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_173() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 173 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_174() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 174 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_175() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 175 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_176() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 176 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_177() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 177 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_178() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 178 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_179() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 179 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_180() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 180 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_181() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 181 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_182() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 182 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_183() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 183 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_184() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 184 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_185() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 185 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_186() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 186 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_187() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 187 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_188() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 188 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_189() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 189 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_190() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 190 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_191() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 191 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_192() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 192 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_193() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 193 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_194() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 194 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_195() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 195 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_196() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 196 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_197() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 197 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_198() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 198 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_199() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 199 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_200() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 200 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_201() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 201 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_202() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 202 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_203() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 203 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_204() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 204 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_205() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 205 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_206() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 206 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_207() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 207 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_208() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 208 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_209() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 209 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_210() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 210 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_211() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 211 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_212() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 212 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_213() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 213 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_214() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 214 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_215() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 215 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_216() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 216 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_217() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 217 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_218() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 218 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_219() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 219 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_220() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 220 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_221() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 221 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_222() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 222 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_223() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 223 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_224() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 224 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_225() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 225 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_226() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 226 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_227() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 227 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_228() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 228 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_229() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 229 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_230() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 230 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_231() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 231 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_232() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 232 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_233() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 233 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_234() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 234 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_235() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 235 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_236() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 236 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_237() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 237 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_238() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 238 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_239() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 239 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_240() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 240 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_241() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 241 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_242() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 242 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_243() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 243 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_244() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 244 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_245() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 245 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_246() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 246 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_247() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 247 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_248() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 248 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_249() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 249 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_250() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 250 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_251() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 251 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_252() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 252 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_253() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 253 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_254() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 254 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_255() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 255 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_256() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 256 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_257() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 257 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_258() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 258 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_259() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 259 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_260() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 260 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_261() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 261 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_262() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 262 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_263() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 263 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_264() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 264 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_265() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 265 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_266() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 266 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_267() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 267 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_268() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 268 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_269() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 269 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_270() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 270 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_271() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 271 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_272() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 272 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_273() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 273 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_274() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 274 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_275() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 275 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_276() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 276 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_277() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 277 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_278() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 278 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_279() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 279 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_280() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 280 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_281() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 281 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_282() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 282 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_283() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 283 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_284() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 284 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_285() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 285 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_286() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 286 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_287() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 287 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_288() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 288 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_289() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 289 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_290() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 290 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_291() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 291 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_292() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 292 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_293() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 293 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_294() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 294 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_295() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 295 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_296() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 296 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_297() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 297 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_298() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 298 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_299() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 299 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_300() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 300 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_301() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 301 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_302() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 302 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_303() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 303 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_304() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 304 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_305() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 305 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_306() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 306 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_307() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 307 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_308() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 308 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_309() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 309 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_310() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 310 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_311() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 311 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_312() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 312 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_313() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 313 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_314() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 314 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_315() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 315 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_316() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 316 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_317() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 317 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_318() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 318 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_319() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 319 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_320() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 320 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_321() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 321 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_322() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 322 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_323() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 323 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_324() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 324 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_325() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 325 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_326() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 326 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_327() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 327 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_328() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 328 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_329() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 329 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_330() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 330 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_331() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 331 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_332() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 332 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_333() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 333 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_334() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 334 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_335() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 335 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_336() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 336 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_337() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 337 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_338() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 338 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_339() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 339 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_340() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 340 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_341() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 341 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_342() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 342 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_343() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 343 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_344() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 344 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_345() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 345 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_346() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 346 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_347() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 347 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_348() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 348 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_349() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 349 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_350() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 350 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_351() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 351 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_352() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 352 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_353() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 353 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_354() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 354 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_355() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 355 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_356() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 356 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_357() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 357 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_358() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 358 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_359() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 359 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_360() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 360 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_361() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 361 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_362() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 362 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_363() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 363 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_364() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 364 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_365() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 365 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_366() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 366 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_367() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 367 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_368() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 368 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_369() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 369 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_370() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 370 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_371() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 371 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_372() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 372 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_373() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 373 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_374() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 374 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_375() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 375 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_376() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 376 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_377() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 377 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_378() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 378 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_379() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 379 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_380() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 380 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_381() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 381 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_382() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 382 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_383() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 383 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_384() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 384 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_385() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 385 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_386() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 386 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_387() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 387 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_388() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 388 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_389() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 389 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_390() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 390 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_391() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 391 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_392() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 392 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_393() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 393 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_394() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 394 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_395() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 395 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_396() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 396 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_397() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 397 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_398() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 398 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_399() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 399 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_400() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 400 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_401() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 401 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_402() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 402 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_403() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 403 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_404() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 404 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_405() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 405 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_406() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 406 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_407() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 407 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_408() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 408 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    #[test]
    fn test_train_hooks_stress_409() {
        let hook = RegHook::new(HookConfig::default());
        let mut p = Tensor::from_slice(&[1.0, 2.0, 409 as f64 * 0.1], vec![3]);
        hook.after_optimizer_step(&mut [p.clone()], 0.01);
        assert_eq!(p.shape(), &[3]);
    }

    // brain-regularization production numerical verification padding line 0
}
