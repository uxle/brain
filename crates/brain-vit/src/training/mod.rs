//! # Training Utilities for brain-vit
//!
//! Provides:
//! - [`Optimizer`] — SGD and Adam optimizers
//! - [`LrScheduler`] — cosine annealing, step decay, warmup
//! - [`Trainer`] — end-to-end training loop for ViT
//! - [`EarlyStopping`] — patience-based stopping criterion
//! - [`GradientClipper`] — gradient clipping by norm

use crate::core::{VitError, VitResult};
use std::collections::HashMap;

/// Supported optimizer types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerType {
    /// Stochastic gradient descent with optional momentum.
    Sgd,
    /// Adam optimizer.
    Adam,
    /// AdamW (Adam with weight decay).
    AdamW,
    /// RMSProp.
    RmsProp,
}

/// Optimizer configuration.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Optimizer variant.
    pub optimizer_type: OptimizerType,
    /// Learning rate.
    pub lr: f64,
    /// Momentum (SGD).
    pub momentum: f64,
    /// Beta1 (Adam).
    pub beta1: f64,
    /// Beta2 (Adam).
    pub beta2: f64,
    /// Epsilon (Adam, RMSProp).
    pub eps: f64,
    /// Weight decay.
    pub weight_decay: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::AdamW,
            lr: 1e-3,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// Optimizer state tracker.
///
/// Stores first and second moment estimates for each parameter group.
pub struct Optimizer {
    /// Configuration.
    pub config: OptimizerConfig,
    /// First moment estimates per parameter name.
    pub m: HashMap<String, Vec<f64>>,
    /// Second moment estimates per parameter name.
    pub v: HashMap<String, Vec<f64>>,
    /// Step count.
    pub step: usize,
}

impl Optimizer {
    /// Create a new optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            v: HashMap::new(),
            step: 0,
        }
    }

    /// Apply one parameter update step.
    ///
    /// - `name`: parameter identifier.
    /// - `params`: current parameter values (updated in-place).
    /// - `grads`: gradient values.
    pub fn step_params(&mut self, name: &str, params: &mut [f64], grads: &[f64]) -> VitResult<()> {
        if params.len() != grads.len() {
            return Err(VitError::Shape(
                "Optimizer: params/grads length mismatch".to_string(),
            ));
        }
        self.step += 1;
        let lr = self.config.lr;
        let wd = self.config.weight_decay;

        match self.config.optimizer_type {
            OptimizerType::Sgd => {
                let mom = self.config.momentum;
                let m = self
                    .m
                    .entry(name.to_string())
                    .or_insert_with(|| vec![0.0; params.len()]);
                for (i, (p, &g)) in params.iter_mut().zip(grads.iter()).enumerate() {
                    m[i] = mom * m[i] + g + wd * *p;
                    *p -= lr * m[i];
                }
            }
            OptimizerType::Adam | OptimizerType::AdamW => {
                let beta1 = self.config.beta1;
                let beta2 = self.config.beta2;
                let eps = self.config.eps;
                let t = self.step as f64;
                let bias_c1 = 1.0 - beta1.powf(t);
                let bias_c2 = 1.0 - beta2.powf(t);
                let lr_t = lr * bias_c2.sqrt() / bias_c1;
                let m = self
                    .m
                    .entry(name.to_string())
                    .or_insert_with(|| vec![0.0; params.len()]);
                let v = self
                    .v
                    .entry(name.to_string())
                    .or_insert_with(|| vec![0.0; params.len()]);
                for (i, (p, &g)) in params.iter_mut().zip(grads.iter()).enumerate() {
                    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
                    let update = lr_t * m[i] / (v[i].sqrt() + eps);
                    if self.config.optimizer_type == OptimizerType::AdamW {
                        *p -= update + lr * wd * *p;
                    } else {
                        *p -= update;
                    }
                }
            }
            OptimizerType::RmsProp => {
                let eps = self.config.eps;
                let rho = self.config.beta2;
                let v = self
                    .v
                    .entry(name.to_string())
                    .or_insert_with(|| vec![0.0; params.len()]);
                for (i, (p, &g)) in params.iter_mut().zip(grads.iter()).enumerate() {
                    v[i] = rho * v[i] + (1.0 - rho) * g * g;
                    *p -= lr * g / (v[i].sqrt() + eps) + lr * wd * *p;
                }
            }
        }
        Ok(())
    }

    /// Zero all momentum buffers.
    pub fn zero_state(&mut self) {
        self.m.clear();
        self.v.clear();
    }

    /// Set learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    /// Current step count.
    pub fn current_step(&self) -> usize {
        self.step
    }
}

/// Learning rate schedule type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScheduleType {
    /// Constant learning rate.
    Constant,
    /// Linear warmup then cosine decay.
    CosineWithWarmup,
    /// Step decay at fixed intervals.
    StepDecay,
    /// Linear decay.
    Linear,
    /// Exponential decay.
    Exponential,
}

/// Learning rate scheduler.
pub struct LrScheduler {
    /// Base (peak) learning rate.
    pub base_lr: f64,
    /// Minimum learning rate.
    pub min_lr: f64,
    /// Total number of training steps.
    pub total_steps: usize,
    /// Warmup steps.
    pub warmup_steps: usize,
    /// Schedule type.
    pub schedule: ScheduleType,
    /// Step decay interval.
    pub step_size: usize,
    /// Step decay gamma.
    pub gamma: f64,
    /// Current step.
    pub current_step: usize,
}

impl LrScheduler {
    /// Create a new scheduler.
    pub fn new(
        base_lr: f64,
        min_lr: f64,
        total_steps: usize,
        warmup_steps: usize,
        schedule: ScheduleType,
    ) -> Self {
        Self {
            base_lr,
            min_lr,
            total_steps,
            warmup_steps,
            schedule,
            step_size: 100,
            gamma: 0.1,
            current_step: 0,
        }
    }

    /// Get the learning rate at the current step.
    pub fn lr(&self) -> f64 {
        let step = self.current_step;
        match self.schedule {
            ScheduleType::Constant => self.base_lr,
            ScheduleType::CosineWithWarmup => {
                if step < self.warmup_steps {
                    self.base_lr * step as f64 / self.warmup_steps.max(1) as f64
                } else {
                    let pi = std::f64::consts::PI;
                    let progress = (step - self.warmup_steps) as f64
                        / (self.total_steps - self.warmup_steps).max(1) as f64;
                    self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (pi * progress).cos())
                }
            }
            ScheduleType::StepDecay => {
                let num_decays = step / self.step_size.max(1);
                self.base_lr * self.gamma.powi(num_decays as i32)
            }
            ScheduleType::Linear => {
                if step >= self.total_steps {
                    return self.min_lr;
                }
                let progress = step as f64 / self.total_steps as f64;
                self.base_lr + (self.min_lr - self.base_lr) * progress
            }
            ScheduleType::Exponential => self.base_lr * self.gamma.powi(step as i32),
        }
    }

    /// Advance the scheduler by one step.
    pub fn step(&mut self) -> f64 {
        let lr = self.lr();
        self.current_step += 1;
        lr
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Early stopping criterion.
pub struct EarlyStopping {
    /// Number of steps with no improvement before stopping.
    pub patience: usize,
    /// Minimum improvement threshold.
    pub min_delta: f64,
    /// Whether to minimize or maximize the metric.
    pub minimize: bool,
    /// Best metric value seen so far.
    pub best: f64,
    /// Steps since last improvement.
    pub wait: usize,
    /// Whether stopping has been triggered.
    pub stopped: bool,
}

impl EarlyStopping {
    /// Create a new early stopping tracker.
    pub fn new(patience: usize, min_delta: f64, minimize: bool) -> Self {
        let best = if minimize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        Self {
            patience,
            min_delta,
            minimize,
            best,
            wait: 0,
            stopped: false,
        }
    }

    /// Update with a new metric value. Returns true if training should stop.
    pub fn update(&mut self, metric: f64) -> bool {
        let improved = if self.minimize {
            metric < self.best - self.min_delta
        } else {
            metric > self.best + self.min_delta
        };
        if improved {
            self.best = metric;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped = true;
            }
        }
        self.stopped
    }

    /// Reset state.
    pub fn reset(&mut self) {
        let best = if self.minimize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        self.best = best;
        self.wait = 0;
        self.stopped = false;
    }
}

/// Gradient clipping by global L2 norm.
pub struct GradientClipper {
    /// Max global gradient norm.
    pub max_norm: f64,
}

impl GradientClipper {
    /// Create a new clipper.
    pub fn new(max_norm: f64) -> VitResult<Self> {
        if max_norm <= 0.0 {
            return Err(VitError::Config(
                "GradientClipper: max_norm must be > 0".to_string(),
            ));
        }
        Ok(Self { max_norm })
    }

    /// Clip gradients in-place, returning the total norm before clipping.
    pub fn clip(&self, grads: &mut [Vec<f64>]) -> f64 {
        let total_sq: f64 = grads.iter().flat_map(|g| g.iter()).map(|&x| x * x).sum();
        let norm = total_sq.sqrt();
        if norm > self.max_norm {
            let scale = self.max_norm / norm;
            for g in grads.iter_mut() {
                for x in g.iter_mut() {
                    *x *= scale;
                }
            }
        }
        norm
    }
}

/// Training metrics for one epoch.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Average training loss.
    pub train_loss: f64,
    /// Training accuracy.
    pub train_acc: f64,
    /// Validation loss.
    pub val_loss: f64,
    /// Validation accuracy.
    pub val_acc: f64,
    /// Learning rate at end of epoch.
    pub lr: f64,
    /// Epoch number (0-indexed).
    pub epoch: usize,
    /// Steps in this epoch.
    pub steps: usize,
}

/// Exponential moving average of model weights.
pub struct ModelEma {
    /// EMA decay factor.
    pub decay: f64,
    /// Shadowed weights per parameter name.
    pub shadow: HashMap<String, Vec<f64>>,
    /// Number of updates.
    pub updates: usize,
}

impl ModelEma {
    /// Create a new EMA tracker.
    pub fn new(decay: f64) -> VitResult<Self> {
        if !(0.0..1.0).contains(&decay) {
            return Err(VitError::Config(
                "ModelEma: decay must be in [0, 1)".to_string(),
            ));
        }
        Ok(Self {
            decay,
            shadow: HashMap::new(),
            updates: 0,
        })
    }

    /// Update shadow weights from current params.
    pub fn update(&mut self, name: &str, params: &[f64]) {
        self.updates += 1;
        // Warm-up: effective decay ramps up in early steps
        let d = self.decay.min(1.0 - 1.0 / (self.updates as f64 + 1.0));
        let shadow = self
            .shadow
            .entry(name.to_string())
            .or_insert_with(|| params.to_vec());
        for (s, &p) in shadow.iter_mut().zip(params.iter()) {
            *s = d * *s + (1.0 - d) * p;
        }
    }

    /// Get shadow weights for a parameter.
    pub fn get(&self, name: &str) -> Option<&Vec<f64>> {
        self.shadow.get(name)
    }
}

/// Mixup data augmentation: linearly interpolates two training samples.
///
/// Returns (mixed_input, mixed_label_a, mixed_label_b, lambda).
pub fn mixup(
    x1: &[f64],
    x2: &[f64],
    y1: usize,
    y2: usize,
    _alpha: f64,
    lambda: f64,
) -> VitResult<(Vec<f64>, usize, usize, f64)> {
    if x1.len() != x2.len() {
        return Err(VitError::Shape(
            "mixup: inputs must have equal length".to_string(),
        ));
    }
    let mixed: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| lambda * a + (1.0 - lambda) * b)
        .collect();
    Ok((mixed, y1, y2, lambda))
}

/// Compute label-smoothed cross-entropy loss.
///
/// `logits`: `[B, C]` flat; `labels`: `[B]` integer class labels.
pub fn label_smooth_ce(
    logits: &[f64],
    labels: &[usize],
    batch: usize,
    num_classes: usize,
    smoothing: f64,
) -> VitResult<f64> {
    if logits.len() != batch * num_classes {
        return Err(VitError::Shape(
            "label_smooth_ce: shape mismatch".to_string(),
        ));
    }
    let eps = smoothing / num_classes as f64;
    let scale = 1.0 - smoothing + eps;
    let mut total = 0.0f64;
    for b in 0..batch {
        let row = &logits[b * num_classes..(b + 1) * num_classes];
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let log_sum_exp = max_val + row.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln();
        let label = labels[b];
        if label >= num_classes {
            return Err(VitError::Shape(format!(
                "label_smooth_ce: label {} >= {}",
                label, num_classes
            )));
        }
        // Hard label component
        let ce_hard = log_sum_exp - row[label];
        // Soft label component (uniform)
        let ce_soft: f64 = row.iter().map(|&x| log_sum_exp - x).sum::<f64>() / num_classes as f64;
        total += scale * ce_hard + eps * ce_soft * num_classes as f64;
    }
    Ok(total / batch as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_sgd_step() {
        let cfg = OptimizerConfig {
            optimizer_type: OptimizerType::Sgd,
            lr: 0.1,
            ..Default::default()
        };
        let mut opt = Optimizer::new(cfg);
        let mut params = vec![1.0f64; 4];
        let grads = vec![1.0f64; 4];
        opt.step_params("w", &mut params, &grads).unwrap();
        for &p in &params {
            assert!(p < 1.0);
        }
    }

    #[test]
    fn test_optimizer_adam_step() {
        let cfg = OptimizerConfig::default();
        let mut opt = Optimizer::new(cfg);
        let mut params = vec![1.0f64; 4];
        let grads = vec![0.5f64; 4];
        opt.step_params("w", &mut params, &grads).unwrap();
        assert_eq!(opt.current_step(), 1);
    }

    #[test]
    fn test_optimizer_shape_mismatch() {
        let cfg = OptimizerConfig::default();
        let mut opt = Optimizer::new(cfg);
        let mut params = vec![1.0f64; 4];
        let grads = vec![0.5f64; 3]; // wrong size
        assert!(opt.step_params("w", &mut params, &grads).is_err());
    }

    #[test]
    fn test_optimizer_zero_state() {
        let cfg = OptimizerConfig::default();
        let mut opt = Optimizer::new(cfg);
        let mut params = vec![1.0f64; 4];
        let grads = vec![0.5f64; 4];
        opt.step_params("w", &mut params, &grads).unwrap();
        opt.zero_state();
        assert!(opt.m.is_empty());
    }

    #[test]
    fn test_optimizer_set_lr() {
        let cfg = OptimizerConfig::default();
        let mut opt = Optimizer::new(cfg);
        opt.set_lr(0.01);
        assert!((opt.config.lr - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_optimizer_rmsprop() {
        let cfg = OptimizerConfig {
            optimizer_type: OptimizerType::RmsProp,
            lr: 0.01,
            ..Default::default()
        };
        let mut opt = Optimizer::new(cfg);
        let mut params = vec![1.0f64; 3];
        let grads = vec![0.1f64; 3];
        opt.step_params("w", &mut params, &grads).unwrap();
        for &p in &params {
            assert!(p < 1.0);
        }
    }

    #[test]
    fn test_scheduler_cosine_warmup() {
        let mut sched = LrScheduler::new(1e-3, 1e-5, 100, 10, ScheduleType::CosineWithWarmup);
        let lr0 = sched.step(); // step 0 in warmup
        assert!(lr0 < 1e-3);
        for _ in 0..10 {
            sched.step();
        } // through warmup
        let lr_peak = sched.step();
        assert!(lr_peak <= 1e-3 + 1e-10);
    }

    #[test]
    fn test_scheduler_constant() {
        let mut sched = LrScheduler::new(1e-3, 0.0, 100, 0, ScheduleType::Constant);
        for _ in 0..10 {
            let lr = sched.step();
            assert!((lr - 1e-3).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scheduler_step_decay() {
        let mut sched = LrScheduler::new(1.0, 0.0, 100, 0, ScheduleType::StepDecay);
        sched.step_size = 5;
        sched.gamma = 0.5;
        for _ in 0..5 {
            sched.step();
        }
        let lr = sched.step();
        assert!(lr < 1.0);
    }

    #[test]
    fn test_scheduler_linear() {
        let mut sched = LrScheduler::new(1.0, 0.0, 10, 0, ScheduleType::Linear);
        let lrs: Vec<f64> = (0..10).map(|_| sched.step()).collect();
        // Should be monotonically decreasing
        for w in lrs.windows(2) {
            assert!(w[0] >= w[1] - 1e-9);
        }
    }

    #[test]
    fn test_scheduler_exponential() {
        let mut sched = LrScheduler::new(1.0, 0.0, 100, 0, ScheduleType::Exponential);
        sched.gamma = 0.9;
        let lr0 = sched.step();
        let lr1 = sched.step();
        assert!(lr1 < lr0);
    }

    #[test]
    fn test_scheduler_reset() {
        let mut sched = LrScheduler::new(1e-3, 1e-5, 100, 10, ScheduleType::CosineWithWarmup);
        for _ in 0..20 {
            sched.step();
        }
        sched.reset();
        assert_eq!(sched.current_step, 0);
    }

    #[test]
    fn test_early_stopping_triggers() {
        let mut es = EarlyStopping::new(3, 1e-4, true);
        assert!(!es.update(1.0));
        assert!(!es.update(1.001)); // no improvement
        assert!(!es.update(1.002)); // no improvement
        assert!(es.update(1.003)); // patience exhausted
    }

    #[test]
    fn test_early_stopping_resets_on_improvement() {
        let mut es = EarlyStopping::new(3, 1e-4, true);
        es.update(1.0);
        es.update(1.001);
        es.update(0.5); // improvement → reset wait
        assert!(!es.update(0.501)); // only 1 bad step
        assert_eq!(es.wait, 1);
    }

    #[test]
    fn test_early_stopping_maximize() {
        let mut es = EarlyStopping::new(2, 0.0, false); // maximize
        assert!(!es.update(0.5));
        assert!(!es.update(0.6)); // improvement
        assert!(!es.update(0.5)); // no improvement
        assert!(es.update(0.5)); // no improvement, patience=2
    }

    #[test]
    fn test_early_stopping_reset() {
        let mut es = EarlyStopping::new(2, 0.0, true);
        es.update(1.0);
        es.update(1.1);
        es.update(1.2);
        assert!(es.stopped);
        es.reset();
        assert!(!es.stopped);
        assert_eq!(es.wait, 0);
    }

    #[test]
    fn test_gradient_clipper_clip() {
        let clipper = GradientClipper::new(1.0).unwrap();
        let mut grads = vec![vec![3.0f64, 4.0f64]]; // norm = 5
        let norm = clipper.clip(&mut grads);
        assert!((norm - 5.0).abs() < 1e-6);
        // After clipping, norm should be ≤ 1.0
        let clipped_norm: f64 = grads
            .iter()
            .flat_map(|g| g.iter())
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt();
        assert!(clipped_norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_gradient_clipper_no_clip_needed() {
        let clipper = GradientClipper::new(10.0).unwrap();
        let mut grads = vec![vec![1.0f64, 0.0f64]];
        let norm = clipper.clip(&mut grads);
        assert!((norm - 1.0).abs() < 1e-9);
        assert!((grads[0][0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_gradient_clipper_zero_norm() {
        let clipper = GradientClipper::new(1.0).unwrap();
        let mut grads = vec![vec![0.0f64; 4]];
        let norm = clipper.clip(&mut grads);
        assert!(norm.abs() < 1e-10);
    }

    #[test]
    fn test_gradient_clipper_invalid() {
        assert!(GradientClipper::new(0.0).is_err());
        assert!(GradientClipper::new(-1.0).is_err());
    }

    #[test]
    fn test_model_ema_update() {
        let mut ema = ModelEma::new(0.9).unwrap();
        let params1 = vec![1.0f64; 4];
        ema.update("w", &params1);
        let params2 = vec![2.0f64; 4];
        ema.update("w", &params2);
        let shadow = ema.get("w").unwrap();
        // Should be between 1.0 and 2.0
        for &s in shadow {
            assert!(s > 1.0 && s < 2.0);
        }
    }

    #[test]
    fn test_model_ema_invalid_decay() {
        assert!(ModelEma::new(1.5).is_err());
        assert!(ModelEma::new(-0.1).is_err());
    }

    #[test]
    fn test_model_ema_get_missing() {
        let ema = ModelEma::new(0.99).unwrap();
        assert!(ema.get("nonexistent").is_none());
    }

    #[test]
    fn test_mixup_output_shape() {
        let x1 = vec![1.0f64; 8];
        let x2 = vec![0.0f64; 8];
        let (mixed, y1, y2, lam) = mixup(&x1, &x2, 3, 5, 0.4, 0.7).unwrap();
        assert_eq!(mixed.len(), 8);
        assert_eq!(y1, 3);
        assert_eq!(y2, 5);
        assert!((lam - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_mixup_value() {
        let x1 = vec![1.0f64];
        let x2 = vec![0.0f64];
        let (mixed, _, _, _) = mixup(&x1, &x2, 0, 1, 0.5, 0.5).unwrap();
        assert!((mixed[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mixup_length_mismatch() {
        assert!(mixup(&[1.0f64; 4], &[0.0f64; 8], 0, 1, 0.5, 0.5).is_err());
    }

    #[test]
    fn test_label_smooth_ce_no_smoothing() {
        // Without smoothing, should match standard CE
        let logits = vec![0.0f64, 100.0, 0.0];
        let labels = vec![1usize];
        let loss = label_smooth_ce(&logits, &labels, 1, 3, 0.0).unwrap();
        assert!(loss < 0.01);
    }

    #[test]
    fn test_label_smooth_ce_with_smoothing() {
        let logits = vec![0.0f64, 100.0, 0.0];
        let labels = vec![1usize];
        let loss_no_smooth = label_smooth_ce(&logits, &labels, 1, 3, 0.0).unwrap();
        let loss_smooth = label_smooth_ce(&logits, &labels, 1, 3, 0.1).unwrap();
        // Smoothing should increase loss slightly on perfect prediction
        assert!(loss_smooth >= loss_no_smooth - 1e-6);
    }

    #[test]
    fn test_label_smooth_ce_invalid_label() {
        let logits = vec![1.0f64; 3];
        let labels = vec![5usize]; // out of range
        assert!(label_smooth_ce(&logits, &labels, 1, 3, 0.1).is_err());
    }

    #[test]
    fn test_training_metrics_default() {
        let m = TrainingMetrics::default();
        assert_eq!(m.train_loss, 0.0);
        assert_eq!(m.epoch, 0);
    }

    #[test]
    fn test_optimizer_adamw_multiple_steps() {
        let cfg = OptimizerConfig::default();
        let mut opt = Optimizer::new(cfg);
        let mut params = vec![0.5f64; 8];
        let grads = vec![0.01f64; 8];
        for _ in 0..10 {
            opt.step_params("w", &mut params, &grads).unwrap();
        }
        assert_eq!(opt.current_step(), 10);
        // All params should have moved
        for &p in &params {
            assert!(p < 0.5);
        }
    }

    #[test]
    fn test_gradient_clipper_multiple_groups() {
        let clipper = GradientClipper::new(1.0).unwrap();
        let mut grads = vec![vec![0.5f64, 0.5f64], vec![0.5f64, 0.5f64]]; // global norm = sqrt(4 * 0.25) = 1.0, no clipping needed
        let norm = clipper.clip(&mut grads);
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ema_updates_count() {
        let mut ema = ModelEma::new(0.9).unwrap();
        for _ in 0..5 {
            ema.update("w", &vec![1.0f64; 4]);
        }
        assert_eq!(ema.updates, 5);
    }
}
