//! # Quantized Model Export Configurations
//!
//! Per-channel scale and zero-point calibration metadata for INT8/UINT8 export.

/// Quantization export configuration.
#[derive(Debug, Clone, Default)]
pub struct QuantExportConfig {
    pub per_channel: bool,
    pub bit_width: usize,
}

impl QuantExportConfig {
    /// Creates a new `QuantExportConfig`.
    pub fn new(per_channel: bool, bit_width: usize) -> Self {
        Self {
            per_channel,
            bit_width,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
