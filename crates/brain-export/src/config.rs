//! # Export Engine Configuration
//!
//! Controls target hardware platform optimizations, graph optimizations, and metadata.

use crate::core::ExportFormat;

/// Target hardware execution platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TargetPlatform {
    #[default]
    Universal,
    MobileArm,
    AppleSilicon,
    WebBrowser,
}

/// Comprehensive model export configuration.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub format: ExportFormat,
    pub platform: TargetPlatform,
    pub optimize_graph: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::default(),
            platform: TargetPlatform::default(),
            optimize_graph: true,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
