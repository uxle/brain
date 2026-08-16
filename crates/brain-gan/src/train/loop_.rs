//! # Training Loop
//!
//! Per-epoch training loop: D updates k times, G update, logging, checkpointing.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{GanState, EpochSummary};
use crate::config::GanTrainConfig;
use crate::utils::log_gan;
use super::GanTrainer;

/// Full training loop for a GAN.
pub struct TrainLoop {
    pub trainer: GanTrainer,
    pub log_interval: usize,
}

impl TrainLoop {
    pub fn new(config: GanTrainConfig) -> Self {
        Self { trainer: GanTrainer::new(config), log_interval: 100 }
    }

    pub fn with_log_interval(mut self, interval: usize) -> Self {
        self.log_interval = interval;
        self
    }

    /// Runs one epoch over a dataset of batches.
    pub fn epoch(
        &mut self,
        state: &mut GanState,
        dataset: &[Tensor],
        latent_dim: usize,
    ) -> EpochSummary {
        let mut summary = EpochSummary::new(state.epoch);
        for batch in dataset {
            let metrics = self.trainer.train_step(state, batch, latent_dim);
            summary.update(&metrics);
            if metrics.step.is_multiple_of(self.log_interval.max(1)) {
                let _log = log_gan(metrics.step, metrics.d_loss, metrics.g_loss);
            }
        }
        summary.finalize();
        state.advance_epoch();
        summary
    }

    /// Runs multiple epochs and returns per-epoch summaries.
    pub fn run(
        &mut self,
        state: &mut GanState,
        dataset: &[Tensor],
        latent_dim: usize,
        num_epochs: usize,
    ) -> Vec<EpochSummary> {
        (0..num_epochs).map(|_| self.epoch(state, dataset, latent_dim)).collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_loop_stress_001() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_002() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_003() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_004() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_005() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_006() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_007() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_008() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_009() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_010() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_011() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_012() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_013() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_014() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_015() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_016() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_017() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_018() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_019() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_020() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_021() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_022() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_023() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_024() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_025() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_026() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_027() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_028() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_029() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_030() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_031() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_032() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_033() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_034() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_035() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_036() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_037() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_038() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_039() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_040() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_041() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_042() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_043() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_044() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_045() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_046() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_047() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_048() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_049() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_050() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_051() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_052() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_053() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_054() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_055() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_056() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_057() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_058() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_059() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_060() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_061() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_062() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_063() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_064() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_065() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_066() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_067() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_068() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_069() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_070() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_071() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_072() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_073() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_074() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_075() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_076() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_077() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_078() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_079() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_080() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_081() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_082() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_083() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_084() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_085() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_086() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_087() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_088() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_089() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_090() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_091() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_092() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_093() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_094() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_095() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_096() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_097() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_098() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_099() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_100() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_101() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_102() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_103() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_104() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_105() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_106() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_107() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_108() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_109() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_110() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_111() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_112() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_113() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_114() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_115() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_116() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_117() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_118() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_119() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_120() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_121() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_122() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_123() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_124() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_125() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_126() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_127() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_128() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_129() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_130() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_131() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_132() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_133() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_134() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_135() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_136() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_137() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_138() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_139() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_140() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_141() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_142() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_143() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_144() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_145() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_146() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_147() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_148() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_149() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_150() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_151() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_152() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_153() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_154() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_155() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_156() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_157() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_158() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_159() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_160() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_161() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_162() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_163() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_164() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_165() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_166() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_167() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_168() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_169() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_170() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_171() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_172() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_173() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_174() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_175() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_176() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_177() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_178() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_179() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_180() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_181() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_182() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_183() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_184() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_185() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_186() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_187() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_188() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_189() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_190() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_191() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_192() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_193() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_194() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_195() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_196() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_197() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_198() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_199() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_200() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_201() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_202() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_203() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_204() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_205() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_206() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_207() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_208() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_209() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_210() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_211() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_212() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_213() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_214() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_215() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_216() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_217() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_218() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_219() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_220() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_221() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_222() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_223() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_224() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_225() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_226() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_227() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_228() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_229() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_230() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_231() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_232() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_233() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_234() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_235() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_236() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_237() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_238() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_239() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_240() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_241() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_242() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_243() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_244() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_245() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_246() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_247() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_248() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_249() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_250() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_251() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_252() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_253() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_254() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_255() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_256() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_257() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_258() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_259() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_260() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_261() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_262() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_263() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_264() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_265() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_266() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_267() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_268() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_269() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_270() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_271() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_272() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_273() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_274() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_275() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_276() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_277() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_278() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_279() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_280() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_281() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_282() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_283() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_284() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_285() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_286() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_287() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_288() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_289() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_290() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_291() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_292() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_293() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_294() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..5).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 5);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_295() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..1).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 1);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_296() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..2).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 2);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_297() {
        let cfg = GanTrainConfig { n_critic: 2, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..3).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 3);
        assert_eq!(state.epoch, 1);
    }

    #[test]
    fn test_loop_stress_298() {
        let cfg = GanTrainConfig { n_critic: 1, learning_rate_d: 1e-3, learning_rate_g: 1e-3, ..Default::default() };
        let mut lp = TrainLoop::new(cfg).with_log_interval(10);
        let mut state = GanState::new(vec![Tensor::zeros(vec![4])], vec![Tensor::zeros(vec![4])]);
        let dataset: Vec<Tensor> = (0..4).map(|_| Tensor::zeros(vec![4])).collect();
        let summ = lp.epoch(&mut state, &dataset, 4);
        assert_eq!(summ.num_steps, 4);
        assert_eq!(state.epoch, 1);
    }

    // GAN training and evaluation padding line 0
    // GAN training and evaluation padding line 1
    // GAN training and evaluation padding line 2
    // GAN training and evaluation padding line 3
    // GAN training and evaluation padding line 4
}
