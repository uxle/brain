//! # ONNX TensorProto Decoder
//!
//! Typed tensor decoding, raw float/double array unpacking, and `brain-core::Tensor` conversion.
#![allow(missing_docs)]

use crate::core::{OnnxError, OnnxResult};
use crate::utils::{read_f32_le, read_f64_le};
use brain_core::Tensor;

/// ONNX Tensor element data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataType {
    Undefined,
    #[default]
    Float,
    Uint8,
    Int8,
    Int32,
    Int64,
    String,
    Bool,
    Float16,
    Double,
}

/// Decoded ONNX TensorProto container.
#[derive(Debug, Clone, Default)]
pub struct TensorProto {
    pub dims: Vec<usize>,
    pub data_type: DataType,
    pub name: String,
    pub raw_data: Vec<u8>,
    pub float_data: Vec<f32>,
    pub double_data: Vec<f64>,
    pub int64_data: Vec<i64>,
}

impl TensorProto {
    pub fn to_tensor(&self) -> OnnxResult<Tensor> {
        let total: usize = self.dims.iter().product();
        let mut f64_vec = Vec::with_capacity(total.max(1));

        if !self.float_data.is_empty() {
            f64_vec.extend(self.float_data.iter().map(|&x| x as f64));
        } else if !self.double_data.is_empty() {
            f64_vec.extend(self.double_data.iter().copied());
        } else if !self.int64_data.is_empty() {
            f64_vec.extend(self.int64_data.iter().map(|&x| x as f64));
        } else if !self.raw_data.is_empty() {
            match self.data_type {
                DataType::Float => {
                    for chunk in self.raw_data.chunks_exact(4) {
                        f64_vec.push(read_f32_le(chunk) as f64);
                    }
                }
                DataType::Double => {
                    for chunk in self.raw_data.chunks_exact(8) {
                        f64_vec.push(read_f64_le(chunk));
                    }
                }
                DataType::Int64 => {
                    for chunk in self.raw_data.chunks_exact(8) {
                        let val = i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        f64_vec.push(val as f64);
                    }
                }
                _ => {
                    return Err(OnnxError::InvalidTensorShape(format!(
                        "Unsupported data type {:?}",
                        self.data_type
                    )));
                }
            }
        } else {
            f64_vec.extend(vec![0.0; total]);
        }

        if f64_vec.len() != total && total > 0 {
            return Err(OnnxError::InvalidTensorShape(format!(
                "Decoded {} elements, expected {}",
                f64_vec.len(),
                total
            )));
        }

        let shape = if self.dims.is_empty() {
            vec![1]
        } else {
            self.dims.clone()
        };
        Ok(Tensor::from_vec(f64_vec, shape))
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
