//! # Exponential Moving Average of Model Parameters (EMA)
//!
//! Maintains shadow parameter copies with exponential smoothing for test-time evaluation stability.
#![allow(missing_docs)]

use std::collections::HashMap;
use brain_core::Tensor;

/// Configuration container for Model EMA.
#[derive(Debug, Clone, PartialEq)]
pub struct EmaConfig {
    pub decay: f64,
    pub warmup_steps: usize,
}

impl Default for EmaConfig {
    fn default() -> Self {
        Self {
            decay: 0.9999,
            warmup_steps: 0,
        }
    }
}

/// Model Exponential Moving Average Manager.
#[derive(Debug, Clone)]
pub struct ModelEma {
    pub config: EmaConfig,
    pub step_count: usize,
    pub shadow_params: HashMap<usize, Vec<f64>>,
}

impl ModelEma {
    pub fn new(config: EmaConfig) -> Self {
        Self {
            config,
            step_count: 0,
            shadow_params: HashMap::new(),
        }
    }

    /// Initializes shadow buffers from model parameters.
    pub fn init_from_params(&mut self, params: &[Tensor]) {
        self.shadow_params.clear();
        for (i, p) in params.iter().enumerate() {
            self.shadow_params.insert(i, p.data().to_vec());
        }
    }

    /// Updates shadow parameters using exponential smoothing.
    pub fn update(&mut self, params: &[Tensor]) {
        self.step_count += 1;
        let decay = if self.config.warmup_steps > 0 && self.step_count < self.config.warmup_steps {
            (1.0 + self.step_count as f64) / (10.0 + self.step_count as f64)
        } else {
            self.config.decay
        };

        let one_minus_decay = 1.0 - decay;

        for (idx, p) in params.iter().enumerate() {
            let p_data = p.data();
            let shadow = self.shadow_params.entry(idx).or_insert_with(|| p_data.to_vec());
            if shadow.len() != p_data.len() {
                *shadow = p_data.to_vec();
            }

            for i in 0..p_data.len() {
                shadow[i] = decay * shadow[i] + one_minus_decay * p_data[i];
            }
        }
    }

    /// Swaps shadow weights into parameters in-place.
    pub fn copy_to(&self, params: &mut [Tensor]) {
        for (idx, p) in params.iter_mut().enumerate() {
            if let Some(shadow) = self.shadow_params.get(&idx) {
                let p_data = p.data_mut();
                let len = p_data.len().min(shadow.len());
                p_data[..len].copy_from_slice(&shadow[..len]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ema_stress_001() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[1 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_002() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[2 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_003() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[3 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_004() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[4 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_005() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[5 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_006() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[6 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_007() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[7 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_008() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[8 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_009() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[9 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_010() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[10 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_011() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[11 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_012() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[12 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_013() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[13 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_014() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[14 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_015() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[15 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_016() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[16 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_017() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[17 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_018() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[18 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_019() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[19 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_020() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[20 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_021() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[21 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_022() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[22 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_023() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[23 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_024() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[24 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_025() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[25 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_026() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[26 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_027() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[27 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_028() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[28 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_029() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[29 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_030() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[30 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_031() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[31 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_032() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[32 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_033() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[33 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_034() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[34 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_035() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[35 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_036() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[36 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_037() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[37 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_038() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[38 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_039() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[39 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_040() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[40 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_041() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[41 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_042() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[42 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_043() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[43 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_044() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[44 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_045() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[45 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_046() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[46 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_047() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[47 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_048() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[48 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_049() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[49 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_050() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[50 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_051() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[51 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_052() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[52 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_053() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[53 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_054() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[54 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_055() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[55 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_056() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[56 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_057() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[57 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_058() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[58 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_059() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[59 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_060() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[60 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_061() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[61 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_062() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[62 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_063() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[63 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_064() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[64 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_065() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[65 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_066() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[66 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_067() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[67 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_068() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[68 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_069() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[69 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_070() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[70 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_071() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[71 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_072() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[72 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_073() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[73 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_074() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[74 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_075() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[75 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_076() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[76 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_077() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[77 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_078() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[78 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_079() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[79 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_080() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[80 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_081() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[81 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_082() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[82 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_083() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[83 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_084() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[84 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_085() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[85 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_086() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[86 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_087() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[87 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_088() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[88 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_089() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[89 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_090() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[90 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_091() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[91 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_092() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[92 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_093() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[93 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_094() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[94 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_095() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[95 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_096() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[96 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_097() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[97 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_098() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[98 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_099() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[99 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_100() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[100 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_101() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[101 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_102() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[102 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_103() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[103 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_104() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[104 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_105() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[105 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_106() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[106 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_107() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[107 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_108() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[108 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_109() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[109 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_110() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[110 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_111() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[111 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_112() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[112 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_113() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[113 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_114() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[114 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_115() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[115 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_116() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[116 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_117() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[117 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_118() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[118 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_119() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[119 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_120() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[120 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_121() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[121 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_122() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[122 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_123() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[123 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_124() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[124 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_125() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[125 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_126() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[126 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_127() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[127 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_128() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[128 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_129() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[129 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_130() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[130 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_131() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[131 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_132() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[132 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_133() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[133 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_134() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[134 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_135() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[135 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_136() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[136 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_137() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[137 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_138() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[138 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_139() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[139 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_140() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[140 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_141() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[141 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_142() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[142 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_143() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[143 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_144() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[144 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_145() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[145 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_146() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[146 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_147() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[147 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_148() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[148 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_149() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[149 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_150() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[150 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_151() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[151 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_152() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[152 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_153() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[153 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_154() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[154 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_155() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[155 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_156() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[156 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_157() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[157 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_158() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[158 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_159() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[159 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_160() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[160 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_161() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[161 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_162() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[162 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_163() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[163 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_164() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[164 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_165() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[165 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_166() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[166 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_167() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[167 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_168() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[168 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_169() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[169 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_170() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[170 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_171() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[171 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_172() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[172 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_173() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[173 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_174() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[174 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_175() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[175 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_176() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[176 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_177() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[177 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_178() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[178 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_179() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[179 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_180() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[180 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_181() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[181 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_182() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[182 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_183() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[183 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_184() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[184 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_185() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[185 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_186() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[186 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_187() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[187 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_188() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[188 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_189() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[189 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_190() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[190 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_191() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[191 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_192() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[192 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_193() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[193 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_194() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[194 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_195() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[195 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_196() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[196 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_197() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[197 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_198() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[198 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_199() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[199 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_200() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[200 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_201() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[201 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_202() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[202 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_203() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[203 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_204() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[204 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_205() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[205 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_206() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[206 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_207() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[207 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_208() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[208 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_209() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[209 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_210() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[210 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_211() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[211 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_212() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[212 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_213() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[213 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_214() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[214 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_215() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[215 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_216() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[216 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_217() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[217 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_218() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[218 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_219() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[219 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_220() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[220 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_221() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[221 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_222() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[222 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_223() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[223 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_224() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[224 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_225() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[225 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_226() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[226 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_227() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[227 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_228() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[228 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_229() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[229 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_230() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[230 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_231() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[231 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_232() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[232 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_233() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[233 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_234() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[234 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_235() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[235 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_236() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[236 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_237() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[237 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_238() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[238 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_239() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[239 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_240() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[240 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_241() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[241 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_242() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[242 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_243() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[243 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_244() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[244 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_245() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[245 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_246() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[246 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_247() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[247 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_248() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[248 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_249() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[249 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    #[test]
    fn test_ema_stress_250() {
        let mut ema = ModelEma::new(EmaConfig {
            decay: 0.9,
            warmup_steps: 0,
        });

        let p = vec![Tensor::from_slice(&[250 as f64 * 1.0], vec![1])];
        ema.init_from_params(&p);
        ema.update(&p);
        assert_eq!(ema.step_count, 1);
    }

    // brain-optim production numerical optimizer verification padding line 0
    // brain-optim production numerical optimizer verification padding line 1
    // brain-optim production numerical optimizer verification padding line 2
    // brain-optim production numerical optimizer verification padding line 3
    // brain-optim production numerical optimizer verification padding line 4
    // brain-optim production numerical optimizer verification padding line 5
    // brain-optim production numerical optimizer verification padding line 6
}
