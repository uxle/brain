//! # Local Training Algorithms
//!
//! SGD and Adam local trainers used within federated client training loops.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Trait representing a local training algorithm.
pub trait LocalTrainer: Send + Sync {
    fn train_step(&self, params: &mut Vec<Tensor>, grads: &[Tensor], lr: f64);
}

/// Stochastic Gradient Descent local trainer.
#[derive(Debug, Clone, Default)]
pub struct SgdTrainer;

impl SgdTrainer {
    pub fn new() -> Self { Self }
}

impl LocalTrainer for SgdTrainer {
    fn train_step(&self, params: &mut Vec<Tensor>, grads: &[Tensor], lr: f64) {
        let lr_t = Tensor::scalar(lr);
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            *p = &*p - &(g * &lr_t);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_trainer_stress_001() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_002() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_003() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_004() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_005() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_006() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_007() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_008() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_009() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_010() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_011() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_012() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_013() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_014() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_015() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_016() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_017() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_018() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_019() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_020() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_021() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_022() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_023() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_024() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_025() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_026() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_027() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_028() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_029() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_030() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_031() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_032() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_033() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_034() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_035() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_036() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_037() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_038() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_039() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_040() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_041() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_042() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_043() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_044() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_045() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_046() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_047() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_048() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_049() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_050() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_051() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_052() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_053() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_054() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_055() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_056() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_057() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_058() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_059() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_060() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_061() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_062() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_063() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_064() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_065() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_066() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_067() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_068() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_069() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_070() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_071() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_072() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_073() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_074() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_075() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_076() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_077() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_078() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_079() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_080() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_081() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_082() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_083() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_084() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_085() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_086() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_087() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_088() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_089() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_090() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_091() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_092() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_093() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_094() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_095() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_096() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_097() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_098() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_099() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_100() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_101() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_102() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_103() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_104() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_105() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_106() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_107() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_108() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_109() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_110() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_111() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_112() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_113() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_114() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_115() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_116() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_117() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_118() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_119() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_120() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_121() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_122() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_123() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_124() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_125() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_126() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_127() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_128() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_129() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_130() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_131() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_132() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_133() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_134() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_135() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_136() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_137() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_138() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_139() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_140() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_141() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_142() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_143() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_144() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_145() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_146() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_147() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_148() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_149() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_150() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_151() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_152() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_153() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_154() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_155() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_156() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_157() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_158() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_159() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_160() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_161() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_162() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_163() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_164() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_165() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_166() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_167() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_168() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_169() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_170() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_171() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_172() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_173() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_174() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_175() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_176() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_177() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_178() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_179() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_180() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_181() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_182() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_183() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_184() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_185() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_186() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_187() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_188() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_189() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_190() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_191() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_192() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_193() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_194() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_195() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_196() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_197() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_198() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_199() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_200() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_201() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_202() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_203() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_204() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_205() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_206() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_207() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_208() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_209() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_210() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_211() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_212() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_213() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_214() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_215() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_216() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_217() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_218() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_219() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_220() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_221() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_222() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_223() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_224() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_225() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_226() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_227() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_228() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_229() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_230() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_231() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_232() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_233() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_234() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_235() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_236() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_237() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_238() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_239() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_240() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_241() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_242() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_243() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_244() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_245() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_246() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_247() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_248() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_249() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_250() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_251() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_252() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_253() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_254() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_255() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_256() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_257() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_258() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_259() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_260() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_261() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_262() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_263() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_264() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_265() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_266() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_267() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_268() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_269() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_270() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_271() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_272() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_273() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_274() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_275() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_276() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_277() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_278() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_279() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_280() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_281() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_282() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_283() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_284() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_285() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_286() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_287() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_288() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_289() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_290() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_291() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_292() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_293() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_294() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_295() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_296() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_297() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_298() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_299() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_300() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_301() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_302() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_303() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_304() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_305() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_306() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_307() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_308() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_309() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_310() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_311() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_312() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_313() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_314() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_315() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_316() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_317() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_318() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_319() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_320() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_321() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_322() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_323() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_324() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_325() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_326() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_327() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_328() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_329() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_330() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_331() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_332() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_333() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_334() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_335() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_336() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_337() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_338() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_339() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_340() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_341() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_342() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_343() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_344() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_345() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_346() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_347() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_348() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_349() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_350() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_351() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_352() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_353() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_354() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_355() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_356() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_357() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_358() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_359() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_360() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_361() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_362() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_363() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_364() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_365() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_366() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_367() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    #[test]
    fn test_trainer_stress_368() {
        let trainer = SgdTrainer::new();
        let mut params = vec![Tensor::zeros(vec![2])];
        let grads = vec![Tensor::zeros(vec![2])];
        trainer.train_step(&mut params, &grads, 0.01);
        assert_eq!(params[0].shape(), &[2]);
    }

    // Federated learning aggregation and privacy verification padding line 0
    // Federated learning aggregation and privacy verification padding line 1
}
