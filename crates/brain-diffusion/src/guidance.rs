//! # Classifier-Free Guidance (CFG) & Thresholding
//!
//! Dynamic thresholding, rescale CFG, and per-step guidance scheduling.

/// Guidance configuration parameters.
#[derive(Debug, Clone)]
pub struct GuidanceConfig {
    pub scale: f64,
    pub dynamic_thresholding: bool,
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            scale: 7.5,
            dynamic_thresholding: false,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
