//! # Teacher Forcing Schedules & Scheduled Sampling
//!
//! Annealing schedules for autoregressive ground-truth token exposure during training.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

/// Scheduled sampling decay schedule.
#[derive(Debug, Clone, PartialEq)]
pub enum TeacherSchedule {
    Constant(f64),
    Linear { start: f64, end: f64, steps: usize },
    Exponential { start: f64, decay_rate: f64, min_ratio: f64 },
}

/// Teacher Forcing Controller.
#[derive(Debug, Clone)]
pub struct TeacherForcer {
    pub schedule: TeacherSchedule,
    pub step: usize,
}

impl TeacherForcer {
    pub fn new(schedule: TeacherSchedule) -> Self {
        Self { schedule, step: 0 }
    }

    pub fn current_ratio(&self) -> f64 {
        match self.schedule {
            TeacherSchedule::Constant(r) => r,
            TeacherSchedule::Linear { start, end, steps } => {
                if self.step >= steps {
                    end
                } else {
                    let p = self.step as f64 / steps as f64;
                    start + (end - start) * p
                }
            }
            TeacherSchedule::Exponential { start, decay_rate, min_ratio } => {
                let r = start * decay_rate.powi(self.step as i32);
                r.max(min_ratio)
            }
        }
    }

    pub fn advance_step(&mut self) {
        self.step += 1;
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::cells::*;
    use crate::seq::*;
    use crate::init_rnn::*;
    use crate::reg_ops::*;
    use crate::process::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::helper::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
