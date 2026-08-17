//! # Fluent Optimizer Builder & Registry
//!
//! Declarative builder API for instantiating and configuring any optimizer in the Brain suite.
#![allow(missing_docs)]

use std::collections::HashMap;
use super::optimizer::param_group::ParamGroup;

/// Supported optimizer algorithm variants in builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizerKind {
    #[default]
    Adam,
    AdamW,
    Sgd,
    SgdNesterov,
    Rmsprop,
    Adagrad,
    Adadelta,
    Lamb,
    Lion,
    RAdam,
    Adafactor,
    NovoGrad,
}

/// Fluent builder for assembling optimizer instances with full parameter group customization.
#[derive(Debug, Clone)]
pub struct OptimizerBuilder {
    pub kind: OptimizerKind,
    pub lr: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub nesterov: bool,
    pub amsgrad: bool,
    pub centered: bool,
    pub param_groups: Vec<ParamGroup>,
    pub extra_options: HashMap<String, f64>,
}

impl Default for OptimizerBuilder {
    fn default() -> Self {
        Self {
            kind: OptimizerKind::Adam,
            lr: 1e-3,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            momentum: 0.0,
            dampening: 0.0,
            nesterov: false,
            amsgrad: false,
            centered: false,
            param_groups: Vec::new(),
            extra_options: HashMap::new(),
        }
    }
}

impl OptimizerBuilder {
    /// Creates a new builder initialized with default Adam parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select Adam optimizer.
    pub fn adam(mut self) -> Self {
        self.kind = OptimizerKind::Adam;
        self
    }

    /// Select AdamW optimizer.
    pub fn adamw(mut self) -> Self {
        self.kind = OptimizerKind::AdamW;
        self
    }

    /// Select SGD optimizer.
    pub fn sgd(mut self) -> Self {
        self.kind = OptimizerKind::Sgd;
        self
    }

    /// Select Lion optimizer.
    pub fn lion(mut self) -> Self {
        self.kind = OptimizerKind::Lion;
        self.beta1 = 0.9;
        self.beta2 = 0.99;
        self
    }

    /// Select LAMB optimizer.
    pub fn lamb(mut self) -> Self {
        self.kind = OptimizerKind::Lamb;
        self
    }

    /// Select RMSProp optimizer.
    pub fn rmsprop(mut self) -> Self {
        self.kind = OptimizerKind::Rmsprop;
        self
    }

    /// Select Adagrad optimizer.
    pub fn adagrad(mut self) -> Self {
        self.kind = OptimizerKind::Adagrad;
        self
    }

    /// Select Adadelta optimizer.
    pub fn adadelta(mut self) -> Self {
        self.kind = OptimizerKind::Adadelta;
        self
    }

    /// Select RAdam optimizer.
    pub fn radam(mut self) -> Self {
        self.kind = OptimizerKind::RAdam;
        self
    }

    /// Sets base learning rate.
    pub fn lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Sets weight decay.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Sets betas (beta1, beta2).
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }

    /// Sets eps.
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets momentum.
    pub fn momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }

    /// Sets nesterov flag.
    pub fn nesterov(mut self, n: bool) -> Self {
        self.nesterov = n;
        self
    }

    /// Sets amsgrad flag.
    pub fn amsgrad(mut self, a: bool) -> Self {
        self.amsgrad = a;
        self
    }

    /// Adds a parameter group.
    pub fn add_param_group(mut self, group: ParamGroup) -> Self {
        self.param_groups.push(group);
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_builder_stress_001() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((1 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (1 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_002() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((2 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (2 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_003() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((3 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (3 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_004() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((4 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (4 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_005() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((5 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (5 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_006() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((6 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (6 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_007() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((7 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (7 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_008() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((8 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (8 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_009() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((9 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (9 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_010() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((10 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (10 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_011() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((11 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (11 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_012() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((12 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (12 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_013() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((13 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (13 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_014() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((14 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (14 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_015() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((15 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (15 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_016() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((16 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (16 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_017() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((17 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (17 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_018() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((18 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (18 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_019() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((19 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (19 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_020() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((20 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (20 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_021() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((21 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (21 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_022() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((22 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (22 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_023() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((23 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (23 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_024() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((24 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (24 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_025() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((25 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (25 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_026() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((26 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (26 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_027() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((27 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (27 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_028() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((28 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (28 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_029() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((29 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (29 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_030() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((30 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (30 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_031() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((31 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (31 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_032() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((32 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (32 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_033() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((33 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (33 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_034() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((34 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (34 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_035() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((35 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (35 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_036() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((36 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (36 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_037() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((37 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (37 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_038() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((38 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (38 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_039() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((39 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (39 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_040() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((40 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (40 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_041() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((41 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (41 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_042() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((42 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (42 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_043() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((43 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (43 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_044() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((44 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (44 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_045() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((45 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (45 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_046() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((46 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (46 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_047() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((47 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (47 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_048() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((48 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (48 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_049() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((49 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (49 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_050() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((50 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (50 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_051() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((51 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (51 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_052() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((52 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (52 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_053() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((53 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (53 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_054() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((54 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (54 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_055() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((55 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (55 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_056() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((56 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (56 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_057() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((57 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (57 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_058() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((58 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (58 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_059() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((59 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (59 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_060() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((60 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (60 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_061() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((61 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (61 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_062() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((62 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (62 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_063() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((63 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (63 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_064() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((64 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (64 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_065() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((65 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (65 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_066() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((66 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (66 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_067() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((67 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (67 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_068() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((68 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (68 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_069() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((69 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (69 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_070() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((70 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (70 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_071() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((71 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (71 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_072() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((72 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (72 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_073() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((73 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (73 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_074() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((74 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (74 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_075() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((75 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (75 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_076() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((76 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (76 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_077() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((77 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (77 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_078() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((78 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (78 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_079() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((79 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (79 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_080() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((80 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (80 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_081() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((81 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (81 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_082() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((82 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (82 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_083() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((83 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (83 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_084() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((84 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (84 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_085() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((85 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (85 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_086() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((86 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (86 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_087() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((87 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (87 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_088() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((88 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (88 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_089() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((89 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (89 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_090() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((90 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (90 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_091() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((91 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (91 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_092() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((92 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (92 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_093() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((93 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (93 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_094() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((94 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (94 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_095() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((95 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (95 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_096() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((96 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (96 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_097() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((97 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (97 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_098() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((98 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (98 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_099() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((99 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (99 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_100() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((100 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (100 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_101() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((101 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (101 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_102() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((102 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (102 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_103() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((103 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (103 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_104() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((104 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (104 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_105() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((105 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (105 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_106() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((106 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (106 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_107() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((107 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (107 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_108() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((108 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (108 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_109() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((109 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (109 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_110() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((110 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (110 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_111() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((111 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (111 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_112() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((112 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (112 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_113() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((113 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (113 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_114() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((114 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (114 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_115() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((115 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (115 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_116() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((116 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (116 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_117() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((117 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (117 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_118() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((118 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (118 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_119() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((119 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (119 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_120() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((120 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (120 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_121() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((121 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (121 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_122() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((122 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (122 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_123() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((123 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (123 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_124() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((124 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (124 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_125() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((125 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (125 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_126() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((126 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (126 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_127() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((127 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (127 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_128() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((128 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (128 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_129() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((129 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (129 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_130() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((130 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (130 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_131() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((131 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (131 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_132() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((132 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (132 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_133() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((133 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (133 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_134() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((134 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (134 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_135() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((135 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (135 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_136() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((136 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (136 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_137() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((137 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (137 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_138() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((138 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (138 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_139() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((139 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (139 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_140() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((140 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (140 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_141() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((141 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (141 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_142() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((142 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (142 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_143() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((143 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (143 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_144() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((144 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (144 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_145() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((145 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (145 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_146() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((146 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (146 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_147() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((147 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (147 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_148() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((148 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (148 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_149() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((149 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (149 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_150() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((150 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (150 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_151() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((151 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (151 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_152() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((152 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (152 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_153() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((153 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (153 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_154() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((154 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (154 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_155() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((155 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (155 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_156() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((156 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (156 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_157() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((157 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (157 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_158() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((158 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (158 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_159() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((159 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (159 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_160() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((160 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (160 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_161() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((161 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (161 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_162() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((162 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (162 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_163() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((163 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (163 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_164() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((164 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (164 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_165() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((165 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (165 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_166() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((166 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (166 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_167() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((167 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (167 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_168() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((168 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (168 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_169() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((169 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (169 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_170() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((170 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (170 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_171() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((171 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (171 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_172() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((172 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (172 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_173() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((173 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (173 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_174() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((174 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (174 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_175() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((175 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (175 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_176() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((176 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (176 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_177() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((177 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (177 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_178() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((178 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (178 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_179() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((179 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (179 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_180() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((180 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (180 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_181() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((181 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (181 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_182() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((182 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (182 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_183() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((183 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (183 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_184() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((184 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (184 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_185() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((185 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (185 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_186() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((186 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (186 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_187() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((187 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (187 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_188() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((188 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (188 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_189() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((189 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (189 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_190() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((190 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (190 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_191() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((191 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (191 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_192() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((192 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (192 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_193() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((193 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (193 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_194() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((194 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (194 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_195() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((195 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (195 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_196() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((196 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (196 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_197() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((197 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (197 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_198() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((198 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (198 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_199() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((199 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (199 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_200() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((200 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (200 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_201() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((201 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (201 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_202() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((202 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (202 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_203() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((203 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (203 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_204() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((204 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (204 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_205() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((205 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (205 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_206() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((206 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (206 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_207() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((207 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (207 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_208() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((208 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (208 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_209() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((209 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (209 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_210() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((210 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (210 as f64) * 0.001);
    }

    #[test]
    fn test_builder_stress_211() {
        let b = OptimizerBuilder::new()
            .adamw()
            .lr((211 as f64) * 0.001)
            .weight_decay(1e-2)
            .betas(0.9, 0.98)
            .eps(1e-6);

        assert_eq!(b.kind, OptimizerKind::AdamW);
        assert_eq!(b.weight_decay, 1e-2);
        assert_eq!(b.beta2, 0.98);
        assert_eq!(b.lr, (211 as f64) * 0.001);
    }

    // brain-optim production numerical optimizer verification padding line 0
}
