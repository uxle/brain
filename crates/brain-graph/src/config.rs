//! # Graph Configuration
//!
//! Optimization levels, verification rigor, and compiler flags.
#![allow(missing_docs)]

/// Optimization pass aggressiveness levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum OptLevel {
    #[default]
    O0, // No optimizations (debug/reference)
    O1, // Dead code elimination & basic constant folding
    O2, // Full CSE, algebraic rewrites, and operator fusion
    O3, // Aggressive layout transforms & in-place memory reuse
}

/// Verification strictness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerificationLevel {
    None,
    Basic,
    #[default]
    Strict,
}

/// Master configuration for graph construction and optimization passes.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    pub opt_level: OptLevel,
    pub verification: VerificationLevel,
    pub enable_cse: bool,
    pub enable_fusion: bool,
    pub enable_const_fold: bool,
    pub enable_dce: bool,
    pub enable_inplace: bool,
    pub max_pass_iterations: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::O2,
            verification: VerificationLevel::Strict,
            enable_cse: true,
            enable_fusion: true,
            enable_const_fold: true,
            enable_dce: true,
            enable_inplace: true,
            max_pass_iterations: 10,
        }
    }
}

impl GraphConfig {
    pub fn for_opt_level(level: OptLevel) -> Self {
        match level {
            OptLevel::O0 => Self {
                opt_level: OptLevel::O0,
                verification: VerificationLevel::Basic,
                enable_cse: false,
                enable_fusion: false,
                enable_const_fold: false,
                enable_dce: false,
                enable_inplace: false,
                max_pass_iterations: 1,
            },
            OptLevel::O1 => Self {
                opt_level: OptLevel::O1,
                verification: VerificationLevel::Strict,
                enable_cse: false,
                enable_fusion: false,
                enable_const_fold: true,
                enable_dce: true,
                enable_inplace: false,
                max_pass_iterations: 3,
            },
            OptLevel::O2 => Self::default(),
            OptLevel::O3 => Self {
                opt_level: OptLevel::O3,
                verification: VerificationLevel::Strict,
                enable_cse: true,
                enable_fusion: true,
                enable_const_fold: true,
                enable_dce: true,
                enable_inplace: true,
                max_pass_iterations: 20,
            },
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.max_pass_iterations == 0 {
            return Err("max_pass_iterations must be at least 1".into());
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "GraphConfig[opt={:?} cse={} fusion={} fold={} dce={} inplace={}]",
            self.opt_level, self.enable_cse, self.enable_fusion,
            self.enable_const_fold, self.enable_dce, self.enable_inplace
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
