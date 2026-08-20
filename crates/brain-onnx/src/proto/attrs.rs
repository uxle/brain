//! # ONNX AttributeProto Handling
//!
//! Typed attribute extraction: Int, Float, String, Tensor, Ints, Floats, Strings, and Tensors.
#![allow(missing_docs)]

use super::tensor::TensorProto;

/// Attribute data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttributeType {
    #[default]
    Undefined,
    Float,
    Int,
    String,
    Tensor,
    Graph,
    Floats,
    Ints,
    Strings,
    Tensors,
}

/// Decoded ONNX AttributeProto.
#[derive(Debug, Clone, Default)]
pub struct AttributeProto {
    pub name: String,
    pub attr_type: AttributeType,
    pub f: f32,
    pub i: i64,
    pub s: String,
    pub t: Option<TensorProto>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<String>,
}

impl AttributeProto {
    pub fn get_int(&self, default: i64) -> i64 {
        if self.attr_type == AttributeType::Int {
            self.i
        } else {
            default
        }
    }

    pub fn get_float(&self, default: f64) -> f64 {
        if self.attr_type == AttributeType::Float {
            self.f as f64
        } else {
            default
        }
    }

    pub fn get_ints(&self) -> &[i64] {
        &self.ints
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
