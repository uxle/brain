//! # Common Intermediate Representation & Helpers
//!
//! Provides the intermediate [`ExportIr`], weight serialization utilities, and data type mappings.

pub mod dtype_map;
pub mod ir;
pub mod weights;

pub use dtype_map::{map_dtype_to_onnx, map_dtype_to_tflite, DTypeKind};
pub use ir::{ExportIr, ExportNode};
pub use weights::serialize_weights_f32;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
