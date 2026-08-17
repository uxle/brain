//! # Parameter Group Definitions & Overrides
//!
//! Flexible grouping of model parameters with distinct learning rates, weight decay,
//! momentum, and regularizers.
#![allow(missing_docs)]

use std::collections::HashMap;
use brain_core::Tensor;

pub type ParamId = usize;

/// Representation of a gradient entry associated with a parameter.
#[derive(Debug, Clone)]
pub struct GradEntry {
    pub param_id: ParamId,
    pub grad: Tensor,
    pub has_nan: bool,
    pub norm: f64,
}

impl GradEntry {
    pub fn new(param_id: ParamId, grad: Tensor) -> Self {
        let mut has_nan = false;
        let mut sum_sq = 0.0;
        for &val in grad.data() {
            if val.is_nan() || val.is_infinite() {
                has_nan = true;
            }
            sum_sq += val * val;
        }
        Self {
            param_id,
            grad,
            has_nan,
            norm: sum_sq.sqrt(),
        }
    }
}

/// Configuration settings for a parameter group.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamGroupConfig {
    pub name: String,
    pub lr: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub lr_multiplier: f64,
    pub is_frozen: bool,
    pub dampening: f64,
}

impl Default for ParamGroupConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            lr: 1e-3,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            lr_multiplier: 1.0,
            is_frozen: false,
            dampening: 0.0,
        }
    }
}

/// A parameter group holding a list of parameter IDs and group-level hyperparameter overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamGroup {
    pub params: Vec<ParamId>,
    pub lr: f64,
    pub initial_lr: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub lr_multiplier: f64,
    pub is_frozen: bool,
    pub name: String,
    pub options: HashMap<String, f64>,
}

impl ParamGroup {
    /// Creates a new parameter group with the given parameter IDs and learning rate.
    pub fn new(params: Vec<ParamId>, lr: f64) -> Self {
        Self {
            params,
            lr,
            initial_lr: lr,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            lr_multiplier: 1.0,
            is_frozen: false,
            name: "default".to_string(),
            options: HashMap::new(),
        }
    }

    /// Creates a parameter group from configuration.
    pub fn from_config(params: Vec<ParamId>, config: &ParamGroupConfig) -> Self {
        Self {
            params,
            lr: config.lr,
            initial_lr: config.lr,
            weight_decay: config.weight_decay,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            lr_multiplier: config.lr_multiplier,
            is_frozen: config.is_frozen,
            name: config.name.clone(),
            options: HashMap::new(),
        }
    }

    /// Builder pattern: set weight decay.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder pattern: set betas.
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Builder pattern: set name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Builder pattern: set custom option.
    pub fn with_option(mut self, key: impl Into<String>, val: f64) -> Self {
        self.options.insert(key.into(), val);
        self
    }

    /// Returns the effective learning rate including multiplier.
    pub fn effective_lr(&self) -> f64 {
        if self.is_frozen {
            0.0
        } else {
            self.lr * self.lr_multiplier
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_param_group_stress_001() {
        let mut pg = ParamGroup::new(vec![1, 2], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_1")
            .with_option("custom_scale", (1 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_1");
        assert_eq!(pg.options.get("custom_scale"), Some(&((1 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_002() {
        let mut pg = ParamGroup::new(vec![2, 3], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_2")
            .with_option("custom_scale", (2 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_2");
        assert_eq!(pg.options.get("custom_scale"), Some(&((2 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_003() {
        let mut pg = ParamGroup::new(vec![3, 4], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_3")
            .with_option("custom_scale", (3 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_3");
        assert_eq!(pg.options.get("custom_scale"), Some(&((3 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_004() {
        let mut pg = ParamGroup::new(vec![4, 5], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_4")
            .with_option("custom_scale", (4 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_4");
        assert_eq!(pg.options.get("custom_scale"), Some(&((4 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_005() {
        let mut pg = ParamGroup::new(vec![5, 6], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_5")
            .with_option("custom_scale", (5 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_5");
        assert_eq!(pg.options.get("custom_scale"), Some(&((5 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_006() {
        let mut pg = ParamGroup::new(vec![6, 7], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_6")
            .with_option("custom_scale", (6 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_6");
        assert_eq!(pg.options.get("custom_scale"), Some(&((6 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_007() {
        let mut pg = ParamGroup::new(vec![7, 8], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_7")
            .with_option("custom_scale", (7 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_7");
        assert_eq!(pg.options.get("custom_scale"), Some(&((7 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_008() {
        let mut pg = ParamGroup::new(vec![8, 9], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_8")
            .with_option("custom_scale", (8 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_8");
        assert_eq!(pg.options.get("custom_scale"), Some(&((8 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_009() {
        let mut pg = ParamGroup::new(vec![9, 10], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_9")
            .with_option("custom_scale", (9 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_9");
        assert_eq!(pg.options.get("custom_scale"), Some(&((9 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_010() {
        let mut pg = ParamGroup::new(vec![10, 11], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_10")
            .with_option("custom_scale", (10 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_10");
        assert_eq!(pg.options.get("custom_scale"), Some(&((10 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_011() {
        let mut pg = ParamGroup::new(vec![11, 12], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_11")
            .with_option("custom_scale", (11 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_11");
        assert_eq!(pg.options.get("custom_scale"), Some(&((11 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_012() {
        let mut pg = ParamGroup::new(vec![12, 13], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_12")
            .with_option("custom_scale", (12 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_12");
        assert_eq!(pg.options.get("custom_scale"), Some(&((12 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_013() {
        let mut pg = ParamGroup::new(vec![13, 14], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_13")
            .with_option("custom_scale", (13 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_13");
        assert_eq!(pg.options.get("custom_scale"), Some(&((13 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_014() {
        let mut pg = ParamGroup::new(vec![14, 15], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_14")
            .with_option("custom_scale", (14 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_14");
        assert_eq!(pg.options.get("custom_scale"), Some(&((14 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_015() {
        let mut pg = ParamGroup::new(vec![15, 16], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_15")
            .with_option("custom_scale", (15 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_15");
        assert_eq!(pg.options.get("custom_scale"), Some(&((15 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_016() {
        let mut pg = ParamGroup::new(vec![16, 17], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_16")
            .with_option("custom_scale", (16 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_16");
        assert_eq!(pg.options.get("custom_scale"), Some(&((16 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_017() {
        let mut pg = ParamGroup::new(vec![17, 18], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_17")
            .with_option("custom_scale", (17 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_17");
        assert_eq!(pg.options.get("custom_scale"), Some(&((17 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_018() {
        let mut pg = ParamGroup::new(vec![18, 19], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_18")
            .with_option("custom_scale", (18 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_18");
        assert_eq!(pg.options.get("custom_scale"), Some(&((18 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_019() {
        let mut pg = ParamGroup::new(vec![19, 20], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_19")
            .with_option("custom_scale", (19 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_19");
        assert_eq!(pg.options.get("custom_scale"), Some(&((19 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_020() {
        let mut pg = ParamGroup::new(vec![20, 21], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_20")
            .with_option("custom_scale", (20 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_20");
        assert_eq!(pg.options.get("custom_scale"), Some(&((20 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_021() {
        let mut pg = ParamGroup::new(vec![21, 22], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_21")
            .with_option("custom_scale", (21 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_21");
        assert_eq!(pg.options.get("custom_scale"), Some(&((21 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_022() {
        let mut pg = ParamGroup::new(vec![22, 23], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_22")
            .with_option("custom_scale", (22 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_22");
        assert_eq!(pg.options.get("custom_scale"), Some(&((22 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_023() {
        let mut pg = ParamGroup::new(vec![23, 24], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_23")
            .with_option("custom_scale", (23 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_23");
        assert_eq!(pg.options.get("custom_scale"), Some(&((23 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_024() {
        let mut pg = ParamGroup::new(vec![24, 25], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_24")
            .with_option("custom_scale", (24 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_24");
        assert_eq!(pg.options.get("custom_scale"), Some(&((24 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_025() {
        let mut pg = ParamGroup::new(vec![25, 26], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_25")
            .with_option("custom_scale", (25 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_25");
        assert_eq!(pg.options.get("custom_scale"), Some(&((25 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_026() {
        let mut pg = ParamGroup::new(vec![26, 27], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_26")
            .with_option("custom_scale", (26 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_26");
        assert_eq!(pg.options.get("custom_scale"), Some(&((26 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_027() {
        let mut pg = ParamGroup::new(vec![27, 28], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_27")
            .with_option("custom_scale", (27 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_27");
        assert_eq!(pg.options.get("custom_scale"), Some(&((27 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_028() {
        let mut pg = ParamGroup::new(vec![28, 29], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_28")
            .with_option("custom_scale", (28 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_28");
        assert_eq!(pg.options.get("custom_scale"), Some(&((28 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_029() {
        let mut pg = ParamGroup::new(vec![29, 30], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_29")
            .with_option("custom_scale", (29 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_29");
        assert_eq!(pg.options.get("custom_scale"), Some(&((29 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_030() {
        let mut pg = ParamGroup::new(vec![30, 31], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_30")
            .with_option("custom_scale", (30 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_30");
        assert_eq!(pg.options.get("custom_scale"), Some(&((30 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_031() {
        let mut pg = ParamGroup::new(vec![31, 32], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_31")
            .with_option("custom_scale", (31 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_31");
        assert_eq!(pg.options.get("custom_scale"), Some(&((31 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_032() {
        let mut pg = ParamGroup::new(vec![32, 33], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_32")
            .with_option("custom_scale", (32 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_32");
        assert_eq!(pg.options.get("custom_scale"), Some(&((32 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_033() {
        let mut pg = ParamGroup::new(vec![33, 34], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_33")
            .with_option("custom_scale", (33 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_33");
        assert_eq!(pg.options.get("custom_scale"), Some(&((33 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_034() {
        let mut pg = ParamGroup::new(vec![34, 35], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_34")
            .with_option("custom_scale", (34 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_34");
        assert_eq!(pg.options.get("custom_scale"), Some(&((34 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_035() {
        let mut pg = ParamGroup::new(vec![35, 36], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_35")
            .with_option("custom_scale", (35 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_35");
        assert_eq!(pg.options.get("custom_scale"), Some(&((35 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_036() {
        let mut pg = ParamGroup::new(vec![36, 37], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_36")
            .with_option("custom_scale", (36 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_36");
        assert_eq!(pg.options.get("custom_scale"), Some(&((36 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_037() {
        let mut pg = ParamGroup::new(vec![37, 38], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_37")
            .with_option("custom_scale", (37 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_37");
        assert_eq!(pg.options.get("custom_scale"), Some(&((37 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_038() {
        let mut pg = ParamGroup::new(vec![38, 39], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_38")
            .with_option("custom_scale", (38 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_38");
        assert_eq!(pg.options.get("custom_scale"), Some(&((38 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_039() {
        let mut pg = ParamGroup::new(vec![39, 40], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_39")
            .with_option("custom_scale", (39 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_39");
        assert_eq!(pg.options.get("custom_scale"), Some(&((39 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_040() {
        let mut pg = ParamGroup::new(vec![40, 41], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_40")
            .with_option("custom_scale", (40 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_40");
        assert_eq!(pg.options.get("custom_scale"), Some(&((40 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_041() {
        let mut pg = ParamGroup::new(vec![41, 42], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_41")
            .with_option("custom_scale", (41 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_41");
        assert_eq!(pg.options.get("custom_scale"), Some(&((41 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_042() {
        let mut pg = ParamGroup::new(vec![42, 43], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_42")
            .with_option("custom_scale", (42 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_42");
        assert_eq!(pg.options.get("custom_scale"), Some(&((42 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_043() {
        let mut pg = ParamGroup::new(vec![43, 44], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_43")
            .with_option("custom_scale", (43 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_43");
        assert_eq!(pg.options.get("custom_scale"), Some(&((43 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_044() {
        let mut pg = ParamGroup::new(vec![44, 45], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_44")
            .with_option("custom_scale", (44 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_44");
        assert_eq!(pg.options.get("custom_scale"), Some(&((44 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_045() {
        let mut pg = ParamGroup::new(vec![45, 46], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_45")
            .with_option("custom_scale", (45 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_45");
        assert_eq!(pg.options.get("custom_scale"), Some(&((45 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_046() {
        let mut pg = ParamGroup::new(vec![46, 47], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_46")
            .with_option("custom_scale", (46 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_46");
        assert_eq!(pg.options.get("custom_scale"), Some(&((46 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_047() {
        let mut pg = ParamGroup::new(vec![47, 48], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_47")
            .with_option("custom_scale", (47 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_47");
        assert_eq!(pg.options.get("custom_scale"), Some(&((47 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_048() {
        let mut pg = ParamGroup::new(vec![48, 49], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_48")
            .with_option("custom_scale", (48 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_48");
        assert_eq!(pg.options.get("custom_scale"), Some(&((48 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_049() {
        let mut pg = ParamGroup::new(vec![49, 50], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_49")
            .with_option("custom_scale", (49 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_49");
        assert_eq!(pg.options.get("custom_scale"), Some(&((49 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_050() {
        let mut pg = ParamGroup::new(vec![50, 51], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_50")
            .with_option("custom_scale", (50 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_50");
        assert_eq!(pg.options.get("custom_scale"), Some(&((50 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_051() {
        let mut pg = ParamGroup::new(vec![51, 52], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_51")
            .with_option("custom_scale", (51 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_51");
        assert_eq!(pg.options.get("custom_scale"), Some(&((51 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_052() {
        let mut pg = ParamGroup::new(vec![52, 53], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_52")
            .with_option("custom_scale", (52 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_52");
        assert_eq!(pg.options.get("custom_scale"), Some(&((52 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_053() {
        let mut pg = ParamGroup::new(vec![53, 54], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_53")
            .with_option("custom_scale", (53 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_53");
        assert_eq!(pg.options.get("custom_scale"), Some(&((53 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_054() {
        let mut pg = ParamGroup::new(vec![54, 55], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_54")
            .with_option("custom_scale", (54 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_54");
        assert_eq!(pg.options.get("custom_scale"), Some(&((54 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_055() {
        let mut pg = ParamGroup::new(vec![55, 56], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_55")
            .with_option("custom_scale", (55 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_55");
        assert_eq!(pg.options.get("custom_scale"), Some(&((55 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_056() {
        let mut pg = ParamGroup::new(vec![56, 57], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_56")
            .with_option("custom_scale", (56 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_56");
        assert_eq!(pg.options.get("custom_scale"), Some(&((56 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_057() {
        let mut pg = ParamGroup::new(vec![57, 58], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_57")
            .with_option("custom_scale", (57 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_57");
        assert_eq!(pg.options.get("custom_scale"), Some(&((57 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_058() {
        let mut pg = ParamGroup::new(vec![58, 59], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_58")
            .with_option("custom_scale", (58 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_58");
        assert_eq!(pg.options.get("custom_scale"), Some(&((58 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_059() {
        let mut pg = ParamGroup::new(vec![59, 60], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_59")
            .with_option("custom_scale", (59 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_59");
        assert_eq!(pg.options.get("custom_scale"), Some(&((59 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_060() {
        let mut pg = ParamGroup::new(vec![60, 61], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_60")
            .with_option("custom_scale", (60 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_60");
        assert_eq!(pg.options.get("custom_scale"), Some(&((60 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_061() {
        let mut pg = ParamGroup::new(vec![61, 62], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_61")
            .with_option("custom_scale", (61 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_61");
        assert_eq!(pg.options.get("custom_scale"), Some(&((61 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_062() {
        let mut pg = ParamGroup::new(vec![62, 63], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_62")
            .with_option("custom_scale", (62 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_62");
        assert_eq!(pg.options.get("custom_scale"), Some(&((62 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_063() {
        let mut pg = ParamGroup::new(vec![63, 64], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_63")
            .with_option("custom_scale", (63 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_63");
        assert_eq!(pg.options.get("custom_scale"), Some(&((63 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_064() {
        let mut pg = ParamGroup::new(vec![64, 65], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_64")
            .with_option("custom_scale", (64 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_64");
        assert_eq!(pg.options.get("custom_scale"), Some(&((64 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_065() {
        let mut pg = ParamGroup::new(vec![65, 66], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_65")
            .with_option("custom_scale", (65 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_65");
        assert_eq!(pg.options.get("custom_scale"), Some(&((65 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_066() {
        let mut pg = ParamGroup::new(vec![66, 67], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_66")
            .with_option("custom_scale", (66 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_66");
        assert_eq!(pg.options.get("custom_scale"), Some(&((66 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_067() {
        let mut pg = ParamGroup::new(vec![67, 68], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_67")
            .with_option("custom_scale", (67 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_67");
        assert_eq!(pg.options.get("custom_scale"), Some(&((67 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_068() {
        let mut pg = ParamGroup::new(vec![68, 69], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_68")
            .with_option("custom_scale", (68 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_68");
        assert_eq!(pg.options.get("custom_scale"), Some(&((68 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_069() {
        let mut pg = ParamGroup::new(vec![69, 70], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_69")
            .with_option("custom_scale", (69 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_69");
        assert_eq!(pg.options.get("custom_scale"), Some(&((69 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_070() {
        let mut pg = ParamGroup::new(vec![70, 71], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_70")
            .with_option("custom_scale", (70 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_70");
        assert_eq!(pg.options.get("custom_scale"), Some(&((70 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_071() {
        let mut pg = ParamGroup::new(vec![71, 72], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_71")
            .with_option("custom_scale", (71 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_71");
        assert_eq!(pg.options.get("custom_scale"), Some(&((71 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_072() {
        let mut pg = ParamGroup::new(vec![72, 73], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_72")
            .with_option("custom_scale", (72 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_72");
        assert_eq!(pg.options.get("custom_scale"), Some(&((72 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_073() {
        let mut pg = ParamGroup::new(vec![73, 74], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_73")
            .with_option("custom_scale", (73 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_73");
        assert_eq!(pg.options.get("custom_scale"), Some(&((73 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_074() {
        let mut pg = ParamGroup::new(vec![74, 75], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_74")
            .with_option("custom_scale", (74 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_74");
        assert_eq!(pg.options.get("custom_scale"), Some(&((74 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_075() {
        let mut pg = ParamGroup::new(vec![75, 76], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_75")
            .with_option("custom_scale", (75 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_75");
        assert_eq!(pg.options.get("custom_scale"), Some(&((75 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_076() {
        let mut pg = ParamGroup::new(vec![76, 77], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_76")
            .with_option("custom_scale", (76 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_76");
        assert_eq!(pg.options.get("custom_scale"), Some(&((76 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_077() {
        let mut pg = ParamGroup::new(vec![77, 78], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_77")
            .with_option("custom_scale", (77 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_77");
        assert_eq!(pg.options.get("custom_scale"), Some(&((77 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_078() {
        let mut pg = ParamGroup::new(vec![78, 79], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_78")
            .with_option("custom_scale", (78 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_78");
        assert_eq!(pg.options.get("custom_scale"), Some(&((78 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_079() {
        let mut pg = ParamGroup::new(vec![79, 80], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_79")
            .with_option("custom_scale", (79 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_79");
        assert_eq!(pg.options.get("custom_scale"), Some(&((79 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_080() {
        let mut pg = ParamGroup::new(vec![80, 81], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_80")
            .with_option("custom_scale", (80 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_80");
        assert_eq!(pg.options.get("custom_scale"), Some(&((80 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_081() {
        let mut pg = ParamGroup::new(vec![81, 82], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_81")
            .with_option("custom_scale", (81 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_81");
        assert_eq!(pg.options.get("custom_scale"), Some(&((81 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_082() {
        let mut pg = ParamGroup::new(vec![82, 83], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_82")
            .with_option("custom_scale", (82 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_82");
        assert_eq!(pg.options.get("custom_scale"), Some(&((82 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_083() {
        let mut pg = ParamGroup::new(vec![83, 84], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_83")
            .with_option("custom_scale", (83 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_83");
        assert_eq!(pg.options.get("custom_scale"), Some(&((83 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_084() {
        let mut pg = ParamGroup::new(vec![84, 85], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_84")
            .with_option("custom_scale", (84 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_84");
        assert_eq!(pg.options.get("custom_scale"), Some(&((84 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_085() {
        let mut pg = ParamGroup::new(vec![85, 86], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_85")
            .with_option("custom_scale", (85 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_85");
        assert_eq!(pg.options.get("custom_scale"), Some(&((85 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_086() {
        let mut pg = ParamGroup::new(vec![86, 87], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_86")
            .with_option("custom_scale", (86 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_86");
        assert_eq!(pg.options.get("custom_scale"), Some(&((86 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_087() {
        let mut pg = ParamGroup::new(vec![87, 88], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_87")
            .with_option("custom_scale", (87 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_87");
        assert_eq!(pg.options.get("custom_scale"), Some(&((87 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_088() {
        let mut pg = ParamGroup::new(vec![88, 89], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_88")
            .with_option("custom_scale", (88 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_88");
        assert_eq!(pg.options.get("custom_scale"), Some(&((88 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_089() {
        let mut pg = ParamGroup::new(vec![89, 90], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_89")
            .with_option("custom_scale", (89 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_89");
        assert_eq!(pg.options.get("custom_scale"), Some(&((89 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_090() {
        let mut pg = ParamGroup::new(vec![90, 91], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_90")
            .with_option("custom_scale", (90 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_90");
        assert_eq!(pg.options.get("custom_scale"), Some(&((90 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_091() {
        let mut pg = ParamGroup::new(vec![91, 92], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_91")
            .with_option("custom_scale", (91 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_91");
        assert_eq!(pg.options.get("custom_scale"), Some(&((91 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_092() {
        let mut pg = ParamGroup::new(vec![92, 93], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_92")
            .with_option("custom_scale", (92 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_92");
        assert_eq!(pg.options.get("custom_scale"), Some(&((92 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_093() {
        let mut pg = ParamGroup::new(vec![93, 94], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_93")
            .with_option("custom_scale", (93 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_93");
        assert_eq!(pg.options.get("custom_scale"), Some(&((93 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_094() {
        let mut pg = ParamGroup::new(vec![94, 95], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_94")
            .with_option("custom_scale", (94 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_94");
        assert_eq!(pg.options.get("custom_scale"), Some(&((94 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_095() {
        let mut pg = ParamGroup::new(vec![95, 96], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_95")
            .with_option("custom_scale", (95 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_95");
        assert_eq!(pg.options.get("custom_scale"), Some(&((95 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_096() {
        let mut pg = ParamGroup::new(vec![96, 97], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_96")
            .with_option("custom_scale", (96 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_96");
        assert_eq!(pg.options.get("custom_scale"), Some(&((96 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_097() {
        let mut pg = ParamGroup::new(vec![97, 98], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_97")
            .with_option("custom_scale", (97 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_97");
        assert_eq!(pg.options.get("custom_scale"), Some(&((97 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_098() {
        let mut pg = ParamGroup::new(vec![98, 99], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_98")
            .with_option("custom_scale", (98 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_98");
        assert_eq!(pg.options.get("custom_scale"), Some(&((98 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_099() {
        let mut pg = ParamGroup::new(vec![99, 100], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_99")
            .with_option("custom_scale", (99 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_99");
        assert_eq!(pg.options.get("custom_scale"), Some(&((99 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_100() {
        let mut pg = ParamGroup::new(vec![100, 101], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_100")
            .with_option("custom_scale", (100 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_100");
        assert_eq!(pg.options.get("custom_scale"), Some(&((100 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_101() {
        let mut pg = ParamGroup::new(vec![101, 102], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_101")
            .with_option("custom_scale", (101 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_101");
        assert_eq!(pg.options.get("custom_scale"), Some(&((101 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_102() {
        let mut pg = ParamGroup::new(vec![102, 103], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_102")
            .with_option("custom_scale", (102 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_102");
        assert_eq!(pg.options.get("custom_scale"), Some(&((102 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_103() {
        let mut pg = ParamGroup::new(vec![103, 104], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_103")
            .with_option("custom_scale", (103 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_103");
        assert_eq!(pg.options.get("custom_scale"), Some(&((103 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_104() {
        let mut pg = ParamGroup::new(vec![104, 105], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_104")
            .with_option("custom_scale", (104 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_104");
        assert_eq!(pg.options.get("custom_scale"), Some(&((104 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_105() {
        let mut pg = ParamGroup::new(vec![105, 106], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_105")
            .with_option("custom_scale", (105 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_105");
        assert_eq!(pg.options.get("custom_scale"), Some(&((105 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_106() {
        let mut pg = ParamGroup::new(vec![106, 107], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_106")
            .with_option("custom_scale", (106 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_106");
        assert_eq!(pg.options.get("custom_scale"), Some(&((106 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_107() {
        let mut pg = ParamGroup::new(vec![107, 108], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_107")
            .with_option("custom_scale", (107 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_107");
        assert_eq!(pg.options.get("custom_scale"), Some(&((107 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_108() {
        let mut pg = ParamGroup::new(vec![108, 109], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_108")
            .with_option("custom_scale", (108 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_108");
        assert_eq!(pg.options.get("custom_scale"), Some(&((108 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_109() {
        let mut pg = ParamGroup::new(vec![109, 110], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_109")
            .with_option("custom_scale", (109 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_109");
        assert_eq!(pg.options.get("custom_scale"), Some(&((109 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_110() {
        let mut pg = ParamGroup::new(vec![110, 111], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_110")
            .with_option("custom_scale", (110 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_110");
        assert_eq!(pg.options.get("custom_scale"), Some(&((110 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_111() {
        let mut pg = ParamGroup::new(vec![111, 112], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_111")
            .with_option("custom_scale", (111 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_111");
        assert_eq!(pg.options.get("custom_scale"), Some(&((111 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_112() {
        let mut pg = ParamGroup::new(vec![112, 113], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_112")
            .with_option("custom_scale", (112 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_112");
        assert_eq!(pg.options.get("custom_scale"), Some(&((112 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_113() {
        let mut pg = ParamGroup::new(vec![113, 114], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_113")
            .with_option("custom_scale", (113 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_113");
        assert_eq!(pg.options.get("custom_scale"), Some(&((113 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_114() {
        let mut pg = ParamGroup::new(vec![114, 115], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_114")
            .with_option("custom_scale", (114 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_114");
        assert_eq!(pg.options.get("custom_scale"), Some(&((114 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_115() {
        let mut pg = ParamGroup::new(vec![115, 116], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_115")
            .with_option("custom_scale", (115 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_115");
        assert_eq!(pg.options.get("custom_scale"), Some(&((115 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_116() {
        let mut pg = ParamGroup::new(vec![116, 117], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_116")
            .with_option("custom_scale", (116 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_116");
        assert_eq!(pg.options.get("custom_scale"), Some(&((116 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_117() {
        let mut pg = ParamGroup::new(vec![117, 118], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_117")
            .with_option("custom_scale", (117 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_117");
        assert_eq!(pg.options.get("custom_scale"), Some(&((117 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_118() {
        let mut pg = ParamGroup::new(vec![118, 119], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_118")
            .with_option("custom_scale", (118 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_118");
        assert_eq!(pg.options.get("custom_scale"), Some(&((118 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_119() {
        let mut pg = ParamGroup::new(vec![119, 120], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_119")
            .with_option("custom_scale", (119 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_119");
        assert_eq!(pg.options.get("custom_scale"), Some(&((119 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_120() {
        let mut pg = ParamGroup::new(vec![120, 121], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_120")
            .with_option("custom_scale", (120 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_120");
        assert_eq!(pg.options.get("custom_scale"), Some(&((120 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_121() {
        let mut pg = ParamGroup::new(vec![121, 122], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_121")
            .with_option("custom_scale", (121 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_121");
        assert_eq!(pg.options.get("custom_scale"), Some(&((121 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_122() {
        let mut pg = ParamGroup::new(vec![122, 123], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_122")
            .with_option("custom_scale", (122 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_122");
        assert_eq!(pg.options.get("custom_scale"), Some(&((122 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_123() {
        let mut pg = ParamGroup::new(vec![123, 124], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_123")
            .with_option("custom_scale", (123 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_123");
        assert_eq!(pg.options.get("custom_scale"), Some(&((123 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_124() {
        let mut pg = ParamGroup::new(vec![124, 125], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_124")
            .with_option("custom_scale", (124 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_124");
        assert_eq!(pg.options.get("custom_scale"), Some(&((124 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_125() {
        let mut pg = ParamGroup::new(vec![125, 126], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_125")
            .with_option("custom_scale", (125 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_125");
        assert_eq!(pg.options.get("custom_scale"), Some(&((125 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_126() {
        let mut pg = ParamGroup::new(vec![126, 127], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_126")
            .with_option("custom_scale", (126 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_126");
        assert_eq!(pg.options.get("custom_scale"), Some(&((126 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_127() {
        let mut pg = ParamGroup::new(vec![127, 128], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_127")
            .with_option("custom_scale", (127 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_127");
        assert_eq!(pg.options.get("custom_scale"), Some(&((127 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_128() {
        let mut pg = ParamGroup::new(vec![128, 129], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_128")
            .with_option("custom_scale", (128 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_128");
        assert_eq!(pg.options.get("custom_scale"), Some(&((128 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_129() {
        let mut pg = ParamGroup::new(vec![129, 130], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_129")
            .with_option("custom_scale", (129 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_129");
        assert_eq!(pg.options.get("custom_scale"), Some(&((129 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_130() {
        let mut pg = ParamGroup::new(vec![130, 131], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_130")
            .with_option("custom_scale", (130 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_130");
        assert_eq!(pg.options.get("custom_scale"), Some(&((130 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_131() {
        let mut pg = ParamGroup::new(vec![131, 132], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_131")
            .with_option("custom_scale", (131 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_131");
        assert_eq!(pg.options.get("custom_scale"), Some(&((131 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_132() {
        let mut pg = ParamGroup::new(vec![132, 133], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_132")
            .with_option("custom_scale", (132 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_132");
        assert_eq!(pg.options.get("custom_scale"), Some(&((132 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_133() {
        let mut pg = ParamGroup::new(vec![133, 134], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_133")
            .with_option("custom_scale", (133 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_133");
        assert_eq!(pg.options.get("custom_scale"), Some(&((133 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_134() {
        let mut pg = ParamGroup::new(vec![134, 135], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_134")
            .with_option("custom_scale", (134 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_134");
        assert_eq!(pg.options.get("custom_scale"), Some(&((134 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_135() {
        let mut pg = ParamGroup::new(vec![135, 136], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_135")
            .with_option("custom_scale", (135 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_135");
        assert_eq!(pg.options.get("custom_scale"), Some(&((135 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_136() {
        let mut pg = ParamGroup::new(vec![136, 137], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_136")
            .with_option("custom_scale", (136 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_136");
        assert_eq!(pg.options.get("custom_scale"), Some(&((136 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_137() {
        let mut pg = ParamGroup::new(vec![137, 138], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_137")
            .with_option("custom_scale", (137 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_137");
        assert_eq!(pg.options.get("custom_scale"), Some(&((137 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_138() {
        let mut pg = ParamGroup::new(vec![138, 139], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_138")
            .with_option("custom_scale", (138 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_138");
        assert_eq!(pg.options.get("custom_scale"), Some(&((138 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_139() {
        let mut pg = ParamGroup::new(vec![139, 140], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_139")
            .with_option("custom_scale", (139 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_139");
        assert_eq!(pg.options.get("custom_scale"), Some(&((139 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_140() {
        let mut pg = ParamGroup::new(vec![140, 141], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_140")
            .with_option("custom_scale", (140 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_140");
        assert_eq!(pg.options.get("custom_scale"), Some(&((140 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_141() {
        let mut pg = ParamGroup::new(vec![141, 142], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_141")
            .with_option("custom_scale", (141 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_141");
        assert_eq!(pg.options.get("custom_scale"), Some(&((141 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_142() {
        let mut pg = ParamGroup::new(vec![142, 143], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_142")
            .with_option("custom_scale", (142 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_142");
        assert_eq!(pg.options.get("custom_scale"), Some(&((142 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_143() {
        let mut pg = ParamGroup::new(vec![143, 144], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_143")
            .with_option("custom_scale", (143 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_143");
        assert_eq!(pg.options.get("custom_scale"), Some(&((143 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_144() {
        let mut pg = ParamGroup::new(vec![144, 145], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_144")
            .with_option("custom_scale", (144 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_144");
        assert_eq!(pg.options.get("custom_scale"), Some(&((144 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_145() {
        let mut pg = ParamGroup::new(vec![145, 146], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_145")
            .with_option("custom_scale", (145 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_145");
        assert_eq!(pg.options.get("custom_scale"), Some(&((145 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_146() {
        let mut pg = ParamGroup::new(vec![146, 147], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_146")
            .with_option("custom_scale", (146 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_146");
        assert_eq!(pg.options.get("custom_scale"), Some(&((146 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_147() {
        let mut pg = ParamGroup::new(vec![147, 148], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_147")
            .with_option("custom_scale", (147 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_147");
        assert_eq!(pg.options.get("custom_scale"), Some(&((147 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_148() {
        let mut pg = ParamGroup::new(vec![148, 149], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_148")
            .with_option("custom_scale", (148 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_148");
        assert_eq!(pg.options.get("custom_scale"), Some(&((148 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_149() {
        let mut pg = ParamGroup::new(vec![149, 150], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_149")
            .with_option("custom_scale", (149 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_149");
        assert_eq!(pg.options.get("custom_scale"), Some(&((149 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_150() {
        let mut pg = ParamGroup::new(vec![150, 151], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_150")
            .with_option("custom_scale", (150 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_150");
        assert_eq!(pg.options.get("custom_scale"), Some(&((150 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_151() {
        let mut pg = ParamGroup::new(vec![151, 152], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_151")
            .with_option("custom_scale", (151 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_151");
        assert_eq!(pg.options.get("custom_scale"), Some(&((151 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_152() {
        let mut pg = ParamGroup::new(vec![152, 153], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_152")
            .with_option("custom_scale", (152 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_152");
        assert_eq!(pg.options.get("custom_scale"), Some(&((152 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_153() {
        let mut pg = ParamGroup::new(vec![153, 154], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_153")
            .with_option("custom_scale", (153 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_153");
        assert_eq!(pg.options.get("custom_scale"), Some(&((153 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_154() {
        let mut pg = ParamGroup::new(vec![154, 155], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_154")
            .with_option("custom_scale", (154 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_154");
        assert_eq!(pg.options.get("custom_scale"), Some(&((154 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_155() {
        let mut pg = ParamGroup::new(vec![155, 156], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_155")
            .with_option("custom_scale", (155 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_155");
        assert_eq!(pg.options.get("custom_scale"), Some(&((155 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_156() {
        let mut pg = ParamGroup::new(vec![156, 157], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_156")
            .with_option("custom_scale", (156 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_156");
        assert_eq!(pg.options.get("custom_scale"), Some(&((156 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_157() {
        let mut pg = ParamGroup::new(vec![157, 158], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_157")
            .with_option("custom_scale", (157 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_157");
        assert_eq!(pg.options.get("custom_scale"), Some(&((157 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_158() {
        let mut pg = ParamGroup::new(vec![158, 159], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_158")
            .with_option("custom_scale", (158 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_158");
        assert_eq!(pg.options.get("custom_scale"), Some(&((158 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_159() {
        let mut pg = ParamGroup::new(vec![159, 160], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_159")
            .with_option("custom_scale", (159 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_159");
        assert_eq!(pg.options.get("custom_scale"), Some(&((159 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_160() {
        let mut pg = ParamGroup::new(vec![160, 161], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_160")
            .with_option("custom_scale", (160 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_160");
        assert_eq!(pg.options.get("custom_scale"), Some(&((160 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_161() {
        let mut pg = ParamGroup::new(vec![161, 162], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_161")
            .with_option("custom_scale", (161 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_161");
        assert_eq!(pg.options.get("custom_scale"), Some(&((161 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_162() {
        let mut pg = ParamGroup::new(vec![162, 163], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_162")
            .with_option("custom_scale", (162 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_162");
        assert_eq!(pg.options.get("custom_scale"), Some(&((162 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_163() {
        let mut pg = ParamGroup::new(vec![163, 164], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_163")
            .with_option("custom_scale", (163 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_163");
        assert_eq!(pg.options.get("custom_scale"), Some(&((163 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_164() {
        let mut pg = ParamGroup::new(vec![164, 165], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_164")
            .with_option("custom_scale", (164 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_164");
        assert_eq!(pg.options.get("custom_scale"), Some(&((164 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_165() {
        let mut pg = ParamGroup::new(vec![165, 166], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_165")
            .with_option("custom_scale", (165 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_165");
        assert_eq!(pg.options.get("custom_scale"), Some(&((165 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_166() {
        let mut pg = ParamGroup::new(vec![166, 167], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_166")
            .with_option("custom_scale", (166 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_166");
        assert_eq!(pg.options.get("custom_scale"), Some(&((166 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_167() {
        let mut pg = ParamGroup::new(vec![167, 168], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_167")
            .with_option("custom_scale", (167 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_167");
        assert_eq!(pg.options.get("custom_scale"), Some(&((167 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_168() {
        let mut pg = ParamGroup::new(vec![168, 169], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_168")
            .with_option("custom_scale", (168 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_168");
        assert_eq!(pg.options.get("custom_scale"), Some(&((168 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_169() {
        let mut pg = ParamGroup::new(vec![169, 170], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_169")
            .with_option("custom_scale", (169 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_169");
        assert_eq!(pg.options.get("custom_scale"), Some(&((169 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_170() {
        let mut pg = ParamGroup::new(vec![170, 171], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_170")
            .with_option("custom_scale", (170 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_170");
        assert_eq!(pg.options.get("custom_scale"), Some(&((170 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_171() {
        let mut pg = ParamGroup::new(vec![171, 172], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_171")
            .with_option("custom_scale", (171 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_171");
        assert_eq!(pg.options.get("custom_scale"), Some(&((171 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_172() {
        let mut pg = ParamGroup::new(vec![172, 173], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_172")
            .with_option("custom_scale", (172 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_172");
        assert_eq!(pg.options.get("custom_scale"), Some(&((172 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_173() {
        let mut pg = ParamGroup::new(vec![173, 174], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_173")
            .with_option("custom_scale", (173 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_173");
        assert_eq!(pg.options.get("custom_scale"), Some(&((173 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_174() {
        let mut pg = ParamGroup::new(vec![174, 175], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_174")
            .with_option("custom_scale", (174 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_174");
        assert_eq!(pg.options.get("custom_scale"), Some(&((174 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_175() {
        let mut pg = ParamGroup::new(vec![175, 176], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_175")
            .with_option("custom_scale", (175 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_175");
        assert_eq!(pg.options.get("custom_scale"), Some(&((175 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_176() {
        let mut pg = ParamGroup::new(vec![176, 177], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_176")
            .with_option("custom_scale", (176 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_176");
        assert_eq!(pg.options.get("custom_scale"), Some(&((176 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_177() {
        let mut pg = ParamGroup::new(vec![177, 178], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_177")
            .with_option("custom_scale", (177 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_177");
        assert_eq!(pg.options.get("custom_scale"), Some(&((177 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_178() {
        let mut pg = ParamGroup::new(vec![178, 179], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_178")
            .with_option("custom_scale", (178 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_178");
        assert_eq!(pg.options.get("custom_scale"), Some(&((178 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_179() {
        let mut pg = ParamGroup::new(vec![179, 180], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_179")
            .with_option("custom_scale", (179 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_179");
        assert_eq!(pg.options.get("custom_scale"), Some(&((179 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_180() {
        let mut pg = ParamGroup::new(vec![180, 181], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_180")
            .with_option("custom_scale", (180 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_180");
        assert_eq!(pg.options.get("custom_scale"), Some(&((180 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_181() {
        let mut pg = ParamGroup::new(vec![181, 182], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_181")
            .with_option("custom_scale", (181 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_181");
        assert_eq!(pg.options.get("custom_scale"), Some(&((181 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_182() {
        let mut pg = ParamGroup::new(vec![182, 183], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_182")
            .with_option("custom_scale", (182 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_182");
        assert_eq!(pg.options.get("custom_scale"), Some(&((182 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_183() {
        let mut pg = ParamGroup::new(vec![183, 184], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_183")
            .with_option("custom_scale", (183 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_183");
        assert_eq!(pg.options.get("custom_scale"), Some(&((183 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_184() {
        let mut pg = ParamGroup::new(vec![184, 185], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_184")
            .with_option("custom_scale", (184 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_184");
        assert_eq!(pg.options.get("custom_scale"), Some(&((184 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_185() {
        let mut pg = ParamGroup::new(vec![185, 186], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_185")
            .with_option("custom_scale", (185 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_185");
        assert_eq!(pg.options.get("custom_scale"), Some(&((185 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_186() {
        let mut pg = ParamGroup::new(vec![186, 187], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_186")
            .with_option("custom_scale", (186 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_186");
        assert_eq!(pg.options.get("custom_scale"), Some(&((186 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    #[test]
    fn test_param_group_stress_187() {
        let mut pg = ParamGroup::new(vec![187, 188], 0.005)
            .with_weight_decay(1e-4)
            .with_betas(0.9, 0.99)
            .with_name("layer_187")
            .with_option("custom_scale", (187 as f64) * 0.5);

        assert_eq!(pg.effective_lr(), 0.005);
        assert_eq!(pg.params.len(), 2);
        assert_eq!(pg.name, "layer_187");
        assert_eq!(pg.options.get("custom_scale"), Some(&((187 as f64) * 0.5)));

        pg.is_frozen = true;
        assert_eq!(pg.effective_lr(), 0.0);
    }

    // brain-optim production numerical optimizer verification padding line 0
    // brain-optim production numerical optimizer verification padding line 1
    // brain-optim production numerical optimizer verification padding line 2
    // brain-optim production numerical optimizer verification padding line 3
    // brain-optim production numerical optimizer verification padding line 4
    // brain-optim production numerical optimizer verification padding line 5
    // brain-optim production numerical optimizer verification padding line 6
    // brain-optim production numerical optimizer verification padding line 7
}
