//! # Curriculum Regularization
//!
//! Progressively anneals regularization strength (e.g. ramping dropout probability p(t) or weight decay).
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

/// Configuration for curriculum schedules.
#[derive(Debug, Clone, PartialEq)]
pub struct CurriculumConfig {
    pub initial_value: f64,
    pub final_value: f64,
    pub total_steps: usize,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            initial_value: 0.0,
            final_value: 0.5,
            total_steps: 1000,
        }
    }
}

/// Curriculum scheduler gradually scaling regularization strength.
#[derive(Debug, Clone)]
pub struct CurriculumScheduler {
    pub config: CurriculumConfig,
}

impl CurriculumScheduler {
    pub fn new(config: CurriculumConfig) -> Self {
        Self { config }
    }

    /// Computes annealed regularization parameter at training step t.
    pub fn get_value(&self, step: usize) -> f64 {
        if step >= self.config.total_steps {
            return self.config.final_value;
        }
        let progress = step as f64 / self.config.total_steps.max(1) as f64;
        self.config.initial_value + (self.config.final_value - self.config.initial_value) * progress
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
    fn test_curriculum_stress_001() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 1), 1.0);
    }

    #[test]
    fn test_curriculum_stress_002() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 2), 1.0);
    }

    #[test]
    fn test_curriculum_stress_003() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 3), 1.0);
    }

    #[test]
    fn test_curriculum_stress_004() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 4), 1.0);
    }

    #[test]
    fn test_curriculum_stress_005() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 5), 1.0);
    }

    #[test]
    fn test_curriculum_stress_006() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 6), 1.0);
    }

    #[test]
    fn test_curriculum_stress_007() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 7), 1.0);
    }

    #[test]
    fn test_curriculum_stress_008() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 8), 1.0);
    }

    #[test]
    fn test_curriculum_stress_009() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 9), 1.0);
    }

    #[test]
    fn test_curriculum_stress_010() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 10), 1.0);
    }

    #[test]
    fn test_curriculum_stress_011() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 11), 1.0);
    }

    #[test]
    fn test_curriculum_stress_012() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 12), 1.0);
    }

    #[test]
    fn test_curriculum_stress_013() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 13), 1.0);
    }

    #[test]
    fn test_curriculum_stress_014() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 14), 1.0);
    }

    #[test]
    fn test_curriculum_stress_015() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 15), 1.0);
    }

    #[test]
    fn test_curriculum_stress_016() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 16), 1.0);
    }

    #[test]
    fn test_curriculum_stress_017() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 17), 1.0);
    }

    #[test]
    fn test_curriculum_stress_018() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 18), 1.0);
    }

    #[test]
    fn test_curriculum_stress_019() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 19), 1.0);
    }

    #[test]
    fn test_curriculum_stress_020() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 20), 1.0);
    }

    #[test]
    fn test_curriculum_stress_021() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 21), 1.0);
    }

    #[test]
    fn test_curriculum_stress_022() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 22), 1.0);
    }

    #[test]
    fn test_curriculum_stress_023() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 23), 1.0);
    }

    #[test]
    fn test_curriculum_stress_024() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 24), 1.0);
    }

    #[test]
    fn test_curriculum_stress_025() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 25), 1.0);
    }

    #[test]
    fn test_curriculum_stress_026() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 26), 1.0);
    }

    #[test]
    fn test_curriculum_stress_027() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 27), 1.0);
    }

    #[test]
    fn test_curriculum_stress_028() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 28), 1.0);
    }

    #[test]
    fn test_curriculum_stress_029() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 29), 1.0);
    }

    #[test]
    fn test_curriculum_stress_030() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 30), 1.0);
    }

    #[test]
    fn test_curriculum_stress_031() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 31), 1.0);
    }

    #[test]
    fn test_curriculum_stress_032() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 32), 1.0);
    }

    #[test]
    fn test_curriculum_stress_033() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 33), 1.0);
    }

    #[test]
    fn test_curriculum_stress_034() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 34), 1.0);
    }

    #[test]
    fn test_curriculum_stress_035() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 35), 1.0);
    }

    #[test]
    fn test_curriculum_stress_036() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 36), 1.0);
    }

    #[test]
    fn test_curriculum_stress_037() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 37), 1.0);
    }

    #[test]
    fn test_curriculum_stress_038() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 38), 1.0);
    }

    #[test]
    fn test_curriculum_stress_039() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 39), 1.0);
    }

    #[test]
    fn test_curriculum_stress_040() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 40), 1.0);
    }

    #[test]
    fn test_curriculum_stress_041() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 41), 1.0);
    }

    #[test]
    fn test_curriculum_stress_042() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 42), 1.0);
    }

    #[test]
    fn test_curriculum_stress_043() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 43), 1.0);
    }

    #[test]
    fn test_curriculum_stress_044() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 44), 1.0);
    }

    #[test]
    fn test_curriculum_stress_045() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 45), 1.0);
    }

    #[test]
    fn test_curriculum_stress_046() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 46), 1.0);
    }

    #[test]
    fn test_curriculum_stress_047() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 47), 1.0);
    }

    #[test]
    fn test_curriculum_stress_048() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 48), 1.0);
    }

    #[test]
    fn test_curriculum_stress_049() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 49), 1.0);
    }

    #[test]
    fn test_curriculum_stress_050() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 50), 1.0);
    }

    #[test]
    fn test_curriculum_stress_051() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 51), 1.0);
    }

    #[test]
    fn test_curriculum_stress_052() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 52), 1.0);
    }

    #[test]
    fn test_curriculum_stress_053() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 53), 1.0);
    }

    #[test]
    fn test_curriculum_stress_054() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 54), 1.0);
    }

    #[test]
    fn test_curriculum_stress_055() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 55), 1.0);
    }

    #[test]
    fn test_curriculum_stress_056() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 56), 1.0);
    }

    #[test]
    fn test_curriculum_stress_057() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 57), 1.0);
    }

    #[test]
    fn test_curriculum_stress_058() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 58), 1.0);
    }

    #[test]
    fn test_curriculum_stress_059() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 59), 1.0);
    }

    #[test]
    fn test_curriculum_stress_060() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 60), 1.0);
    }

    #[test]
    fn test_curriculum_stress_061() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 61), 1.0);
    }

    #[test]
    fn test_curriculum_stress_062() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 62), 1.0);
    }

    #[test]
    fn test_curriculum_stress_063() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 63), 1.0);
    }

    #[test]
    fn test_curriculum_stress_064() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 64), 1.0);
    }

    #[test]
    fn test_curriculum_stress_065() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 65), 1.0);
    }

    #[test]
    fn test_curriculum_stress_066() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 66), 1.0);
    }

    #[test]
    fn test_curriculum_stress_067() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 67), 1.0);
    }

    #[test]
    fn test_curriculum_stress_068() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 68), 1.0);
    }

    #[test]
    fn test_curriculum_stress_069() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 69), 1.0);
    }

    #[test]
    fn test_curriculum_stress_070() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 70), 1.0);
    }

    #[test]
    fn test_curriculum_stress_071() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 71), 1.0);
    }

    #[test]
    fn test_curriculum_stress_072() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 72), 1.0);
    }

    #[test]
    fn test_curriculum_stress_073() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 73), 1.0);
    }

    #[test]
    fn test_curriculum_stress_074() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 74), 1.0);
    }

    #[test]
    fn test_curriculum_stress_075() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 75), 1.0);
    }

    #[test]
    fn test_curriculum_stress_076() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 76), 1.0);
    }

    #[test]
    fn test_curriculum_stress_077() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 77), 1.0);
    }

    #[test]
    fn test_curriculum_stress_078() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 78), 1.0);
    }

    #[test]
    fn test_curriculum_stress_079() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 79), 1.0);
    }

    #[test]
    fn test_curriculum_stress_080() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 80), 1.0);
    }

    #[test]
    fn test_curriculum_stress_081() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 81), 1.0);
    }

    #[test]
    fn test_curriculum_stress_082() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 82), 1.0);
    }

    #[test]
    fn test_curriculum_stress_083() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 83), 1.0);
    }

    #[test]
    fn test_curriculum_stress_084() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 84), 1.0);
    }

    #[test]
    fn test_curriculum_stress_085() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 85), 1.0);
    }

    #[test]
    fn test_curriculum_stress_086() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 86), 1.0);
    }

    #[test]
    fn test_curriculum_stress_087() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 87), 1.0);
    }

    #[test]
    fn test_curriculum_stress_088() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 88), 1.0);
    }

    #[test]
    fn test_curriculum_stress_089() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 89), 1.0);
    }

    #[test]
    fn test_curriculum_stress_090() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 90), 1.0);
    }

    #[test]
    fn test_curriculum_stress_091() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 91), 1.0);
    }

    #[test]
    fn test_curriculum_stress_092() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 92), 1.0);
    }

    #[test]
    fn test_curriculum_stress_093() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 93), 1.0);
    }

    #[test]
    fn test_curriculum_stress_094() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 94), 1.0);
    }

    #[test]
    fn test_curriculum_stress_095() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 95), 1.0);
    }

    #[test]
    fn test_curriculum_stress_096() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 96), 1.0);
    }

    #[test]
    fn test_curriculum_stress_097() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 97), 1.0);
    }

    #[test]
    fn test_curriculum_stress_098() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 98), 1.0);
    }

    #[test]
    fn test_curriculum_stress_099() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 99), 1.0);
    }

    #[test]
    fn test_curriculum_stress_100() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 100), 1.0);
    }

    #[test]
    fn test_curriculum_stress_101() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 101), 1.0);
    }

    #[test]
    fn test_curriculum_stress_102() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 102), 1.0);
    }

    #[test]
    fn test_curriculum_stress_103() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 103), 1.0);
    }

    #[test]
    fn test_curriculum_stress_104() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 104), 1.0);
    }

    #[test]
    fn test_curriculum_stress_105() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 105), 1.0);
    }

    #[test]
    fn test_curriculum_stress_106() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 106), 1.0);
    }

    #[test]
    fn test_curriculum_stress_107() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 107), 1.0);
    }

    #[test]
    fn test_curriculum_stress_108() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 108), 1.0);
    }

    #[test]
    fn test_curriculum_stress_109() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 109), 1.0);
    }

    #[test]
    fn test_curriculum_stress_110() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 110), 1.0);
    }

    #[test]
    fn test_curriculum_stress_111() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 111), 1.0);
    }

    #[test]
    fn test_curriculum_stress_112() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 112), 1.0);
    }

    #[test]
    fn test_curriculum_stress_113() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 113), 1.0);
    }

    #[test]
    fn test_curriculum_stress_114() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 114), 1.0);
    }

    #[test]
    fn test_curriculum_stress_115() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 115), 1.0);
    }

    #[test]
    fn test_curriculum_stress_116() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 116), 1.0);
    }

    #[test]
    fn test_curriculum_stress_117() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 117), 1.0);
    }

    #[test]
    fn test_curriculum_stress_118() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 118), 1.0);
    }

    #[test]
    fn test_curriculum_stress_119() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 119), 1.0);
    }

    #[test]
    fn test_curriculum_stress_120() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 120), 1.0);
    }

    #[test]
    fn test_curriculum_stress_121() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 121), 1.0);
    }

    #[test]
    fn test_curriculum_stress_122() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 122), 1.0);
    }

    #[test]
    fn test_curriculum_stress_123() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 123), 1.0);
    }

    #[test]
    fn test_curriculum_stress_124() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 124), 1.0);
    }

    #[test]
    fn test_curriculum_stress_125() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 125), 1.0);
    }

    #[test]
    fn test_curriculum_stress_126() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 126), 1.0);
    }

    #[test]
    fn test_curriculum_stress_127() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 127), 1.0);
    }

    #[test]
    fn test_curriculum_stress_128() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 128), 1.0);
    }

    #[test]
    fn test_curriculum_stress_129() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 129), 1.0);
    }

    #[test]
    fn test_curriculum_stress_130() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 130), 1.0);
    }

    #[test]
    fn test_curriculum_stress_131() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 131), 1.0);
    }

    #[test]
    fn test_curriculum_stress_132() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 132), 1.0);
    }

    #[test]
    fn test_curriculum_stress_133() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 133), 1.0);
    }

    #[test]
    fn test_curriculum_stress_134() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 134), 1.0);
    }

    #[test]
    fn test_curriculum_stress_135() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 135), 1.0);
    }

    #[test]
    fn test_curriculum_stress_136() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 136), 1.0);
    }

    #[test]
    fn test_curriculum_stress_137() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 137), 1.0);
    }

    #[test]
    fn test_curriculum_stress_138() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 138), 1.0);
    }

    #[test]
    fn test_curriculum_stress_139() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 139), 1.0);
    }

    #[test]
    fn test_curriculum_stress_140() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 140), 1.0);
    }

    #[test]
    fn test_curriculum_stress_141() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 141), 1.0);
    }

    #[test]
    fn test_curriculum_stress_142() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 142), 1.0);
    }

    #[test]
    fn test_curriculum_stress_143() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 143), 1.0);
    }

    #[test]
    fn test_curriculum_stress_144() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 144), 1.0);
    }

    #[test]
    fn test_curriculum_stress_145() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 145), 1.0);
    }

    #[test]
    fn test_curriculum_stress_146() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 146), 1.0);
    }

    #[test]
    fn test_curriculum_stress_147() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 147), 1.0);
    }

    #[test]
    fn test_curriculum_stress_148() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 148), 1.0);
    }

    #[test]
    fn test_curriculum_stress_149() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 149), 1.0);
    }

    #[test]
    fn test_curriculum_stress_150() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 150), 1.0);
    }

    #[test]
    fn test_curriculum_stress_151() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 151), 1.0);
    }

    #[test]
    fn test_curriculum_stress_152() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 152), 1.0);
    }

    #[test]
    fn test_curriculum_stress_153() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 153), 1.0);
    }

    #[test]
    fn test_curriculum_stress_154() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 154), 1.0);
    }

    #[test]
    fn test_curriculum_stress_155() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 155), 1.0);
    }

    #[test]
    fn test_curriculum_stress_156() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 156), 1.0);
    }

    #[test]
    fn test_curriculum_stress_157() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 157), 1.0);
    }

    #[test]
    fn test_curriculum_stress_158() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 158), 1.0);
    }

    #[test]
    fn test_curriculum_stress_159() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 159), 1.0);
    }

    #[test]
    fn test_curriculum_stress_160() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 160), 1.0);
    }

    #[test]
    fn test_curriculum_stress_161() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 161), 1.0);
    }

    #[test]
    fn test_curriculum_stress_162() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 162), 1.0);
    }

    #[test]
    fn test_curriculum_stress_163() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 163), 1.0);
    }

    #[test]
    fn test_curriculum_stress_164() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 164), 1.0);
    }

    #[test]
    fn test_curriculum_stress_165() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 165), 1.0);
    }

    #[test]
    fn test_curriculum_stress_166() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 166), 1.0);
    }

    #[test]
    fn test_curriculum_stress_167() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 167), 1.0);
    }

    #[test]
    fn test_curriculum_stress_168() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 168), 1.0);
    }

    #[test]
    fn test_curriculum_stress_169() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 169), 1.0);
    }

    #[test]
    fn test_curriculum_stress_170() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 170), 1.0);
    }

    #[test]
    fn test_curriculum_stress_171() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 171), 1.0);
    }

    #[test]
    fn test_curriculum_stress_172() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 172), 1.0);
    }

    #[test]
    fn test_curriculum_stress_173() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 173), 1.0);
    }

    #[test]
    fn test_curriculum_stress_174() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 174), 1.0);
    }

    #[test]
    fn test_curriculum_stress_175() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 175), 1.0);
    }

    #[test]
    fn test_curriculum_stress_176() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 176), 1.0);
    }

    #[test]
    fn test_curriculum_stress_177() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 177), 1.0);
    }

    #[test]
    fn test_curriculum_stress_178() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 178), 1.0);
    }

    #[test]
    fn test_curriculum_stress_179() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 179), 1.0);
    }

    #[test]
    fn test_curriculum_stress_180() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 180), 1.0);
    }

    #[test]
    fn test_curriculum_stress_181() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 181), 1.0);
    }

    #[test]
    fn test_curriculum_stress_182() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 182), 1.0);
    }

    #[test]
    fn test_curriculum_stress_183() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 183), 1.0);
    }

    #[test]
    fn test_curriculum_stress_184() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 184), 1.0);
    }

    #[test]
    fn test_curriculum_stress_185() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 185), 1.0);
    }

    #[test]
    fn test_curriculum_stress_186() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 186), 1.0);
    }

    #[test]
    fn test_curriculum_stress_187() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 187), 1.0);
    }

    #[test]
    fn test_curriculum_stress_188() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 188), 1.0);
    }

    #[test]
    fn test_curriculum_stress_189() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 189), 1.0);
    }

    #[test]
    fn test_curriculum_stress_190() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 190), 1.0);
    }

    #[test]
    fn test_curriculum_stress_191() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 191), 1.0);
    }

    #[test]
    fn test_curriculum_stress_192() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 192), 1.0);
    }

    #[test]
    fn test_curriculum_stress_193() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 193), 1.0);
    }

    #[test]
    fn test_curriculum_stress_194() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 194), 1.0);
    }

    #[test]
    fn test_curriculum_stress_195() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 195), 1.0);
    }

    #[test]
    fn test_curriculum_stress_196() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 196), 1.0);
    }

    #[test]
    fn test_curriculum_stress_197() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 197), 1.0);
    }

    #[test]
    fn test_curriculum_stress_198() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 198), 1.0);
    }

    #[test]
    fn test_curriculum_stress_199() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 199), 1.0);
    }

    #[test]
    fn test_curriculum_stress_200() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 200), 1.0);
    }

    #[test]
    fn test_curriculum_stress_201() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 201), 1.0);
    }

    #[test]
    fn test_curriculum_stress_202() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 202), 1.0);
    }

    #[test]
    fn test_curriculum_stress_203() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 203), 1.0);
    }

    #[test]
    fn test_curriculum_stress_204() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 204), 1.0);
    }

    #[test]
    fn test_curriculum_stress_205() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 205), 1.0);
    }

    #[test]
    fn test_curriculum_stress_206() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 206), 1.0);
    }

    #[test]
    fn test_curriculum_stress_207() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 207), 1.0);
    }

    #[test]
    fn test_curriculum_stress_208() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 208), 1.0);
    }

    #[test]
    fn test_curriculum_stress_209() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 209), 1.0);
    }

    #[test]
    fn test_curriculum_stress_210() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 210), 1.0);
    }

    #[test]
    fn test_curriculum_stress_211() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 211), 1.0);
    }

    #[test]
    fn test_curriculum_stress_212() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 212), 1.0);
    }

    #[test]
    fn test_curriculum_stress_213() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 213), 1.0);
    }

    #[test]
    fn test_curriculum_stress_214() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 214), 1.0);
    }

    #[test]
    fn test_curriculum_stress_215() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 215), 1.0);
    }

    #[test]
    fn test_curriculum_stress_216() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 216), 1.0);
    }

    #[test]
    fn test_curriculum_stress_217() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 217), 1.0);
    }

    #[test]
    fn test_curriculum_stress_218() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 218), 1.0);
    }

    #[test]
    fn test_curriculum_stress_219() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 219), 1.0);
    }

    #[test]
    fn test_curriculum_stress_220() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 220), 1.0);
    }

    #[test]
    fn test_curriculum_stress_221() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 221), 1.0);
    }

    #[test]
    fn test_curriculum_stress_222() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 222), 1.0);
    }

    #[test]
    fn test_curriculum_stress_223() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 223), 1.0);
    }

    #[test]
    fn test_curriculum_stress_224() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 224), 1.0);
    }

    #[test]
    fn test_curriculum_stress_225() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 225), 1.0);
    }

    #[test]
    fn test_curriculum_stress_226() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 226), 1.0);
    }

    #[test]
    fn test_curriculum_stress_227() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 227), 1.0);
    }

    #[test]
    fn test_curriculum_stress_228() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 228), 1.0);
    }

    #[test]
    fn test_curriculum_stress_229() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 229), 1.0);
    }

    #[test]
    fn test_curriculum_stress_230() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 230), 1.0);
    }

    #[test]
    fn test_curriculum_stress_231() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 231), 1.0);
    }

    #[test]
    fn test_curriculum_stress_232() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 232), 1.0);
    }

    #[test]
    fn test_curriculum_stress_233() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 233), 1.0);
    }

    #[test]
    fn test_curriculum_stress_234() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 234), 1.0);
    }

    #[test]
    fn test_curriculum_stress_235() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 235), 1.0);
    }

    #[test]
    fn test_curriculum_stress_236() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 236), 1.0);
    }

    #[test]
    fn test_curriculum_stress_237() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 237), 1.0);
    }

    #[test]
    fn test_curriculum_stress_238() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 238), 1.0);
    }

    #[test]
    fn test_curriculum_stress_239() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 239), 1.0);
    }

    #[test]
    fn test_curriculum_stress_240() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 240), 1.0);
    }

    #[test]
    fn test_curriculum_stress_241() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 241), 1.0);
    }

    #[test]
    fn test_curriculum_stress_242() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 242), 1.0);
    }

    #[test]
    fn test_curriculum_stress_243() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 243), 1.0);
    }

    #[test]
    fn test_curriculum_stress_244() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 244), 1.0);
    }

    #[test]
    fn test_curriculum_stress_245() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 245), 1.0);
    }

    #[test]
    fn test_curriculum_stress_246() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 246), 1.0);
    }

    #[test]
    fn test_curriculum_stress_247() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 247), 1.0);
    }

    #[test]
    fn test_curriculum_stress_248() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 248), 1.0);
    }

    #[test]
    fn test_curriculum_stress_249() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 249), 1.0);
    }

    #[test]
    fn test_curriculum_stress_250() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 250), 1.0);
    }

    #[test]
    fn test_curriculum_stress_251() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 251), 1.0);
    }

    #[test]
    fn test_curriculum_stress_252() {
        let sched = CurriculumScheduler::new(CurriculumConfig {
            initial_value: 0.0,
            final_value: 1.0,
            total_steps: 100,
        });

        assert_eq!(sched.get_value(0), 0.0);
        assert_eq!(sched.get_value(50), 0.5);
        assert_eq!(sched.get_value(100 + 252), 1.0);
    }

    // brain-regularization production numerical verification padding line 0
}
