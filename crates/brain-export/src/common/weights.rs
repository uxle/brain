//! # Weight Serialization & DType Quantization
//!
//! Flattens tensor weights into target format byte buffers with optional 32-bit floating point downcasting.

use brain_core::Tensor;

/// Converts a 64-bit tensor weight to 32-bit little-endian byte array.
pub fn serialize_weights_f32(tensor: &Tensor) -> Vec<u8> {
    let mut out = Vec::with_capacity(tensor.numel() * 4);
    for &val in &tensor.to_vec() {
        let val32 = val as f32;
        out.extend_from_slice(&val32.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
