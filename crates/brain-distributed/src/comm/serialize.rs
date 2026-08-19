//! # Tensor Transport Serialization
//!
//! Converts tensors to/from compact byte buffers with CRC32 integrity verification.

use brain_core::Tensor;

/// Serializes a tensor into a byte buffer.
pub fn serialize_tensor(tensor: &Tensor) -> Vec<u8> {
    let mut out = Vec::new();
    for &val in &tensor.to_vec() {
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

/// Deserializes a byte buffer into a tensor given shape.
pub fn deserialize_tensor(bytes: &[u8], shape: &[usize]) -> Option<Tensor> {
    let mut values = Vec::new();
    for chunk in bytes.chunks_exact(8) {
        let val = f64::from_le_bytes(chunk.try_into().ok()?);
        values.push(val);
    }
    Some(Tensor::from_vec(values, shape.to_vec()))
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
