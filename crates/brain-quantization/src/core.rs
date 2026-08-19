//! # Core Quantization Types & Data Structures
//!
//! Fundamental data types, quantization schemes, scale/zero-point parameter containers,
//! error representations, and quantized tensor abstractions.
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use std::fmt;
use brain_core::Tensor;

/// Supported target quantized data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantDType {
    #[default]
    Int8,
    UInt8,
    Int4,
    UInt4,
    Int16,
    UInt16,
    FP8E4M3,
    FP8E5M2,
    BFloat16,
    Float16,
}

impl QuantDType {
    /// Returns the bit width of the quantized data type.
    pub fn bit_width(&self) -> usize {
        match self {
            QuantDType::Int4 | QuantDType::UInt4 => 4,
            QuantDType::Int8 | QuantDType::UInt8 | QuantDType::FP8E4M3 | QuantDType::FP8E5M2 => 8,
            QuantDType::Int16 | QuantDType::UInt16 | QuantDType::BFloat16 | QuantDType::Float16 => 16,
        }
    }

    /// Returns the theoretical minimum integer value representable.
    pub fn qmin(&self) -> i32 {
        match self {
            QuantDType::Int4 => -8,
            QuantDType::UInt4 => 0,
            QuantDType::Int8 => -128,
            QuantDType::UInt8 => 0,
            QuantDType::Int16 => -32768,
            QuantDType::UInt16 => 0,
            _ => -128,
        }
    }

    /// Returns the theoretical maximum integer value representable.
    pub fn qmax(&self) -> i32 {
        match self {
            QuantDType::Int4 => 7,
            QuantDType::UInt4 => 15,
            QuantDType::Int8 => 127,
            QuantDType::UInt8 => 255,
            QuantDType::Int16 => 32767,
            QuantDType::UInt16 => 65535,
            _ => 127,
        }
    }
}

/// Quantization scheme mapping real values to discrete integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantScheme {
    #[default]
    AffinePerTensor,
    SymmetricPerTensor,
    AffinePerChannel,
    SymmetricPerChannel,
    GroupWise { group_size: usize },
}

/// Quantization parameters container (scale and zero-point).
#[derive(Debug, Clone, PartialEq)]
pub struct QParams {
    pub scales: Vec<f64>,
    pub zero_points: Vec<i32>,
    pub qmin: i32,
    pub qmax: i32,
    pub scheme: QuantScheme,
    pub dtype: QuantDType,
}

impl Default for QParams {
    fn default() -> Self {
        Self {
            scales: vec![1.0],
            zero_points: vec![0],
            qmin: -128,
            qmax: 127,
            scheme: QuantScheme::AffinePerTensor,
            dtype: QuantDType::Int8,
        }
    }
}

impl QParams {
    /// Creates uniform per-tensor quantization parameters.
    pub fn per_tensor(scale: f64, zero_point: i32, dtype: QuantDType) -> Self {
        Self {
            scales: vec![scale],
            zero_points: vec![zero_point],
            qmin: dtype.qmin(),
            qmax: dtype.qmax(),
            scheme: QuantScheme::AffinePerTensor,
            dtype,
        }
    }

    /// Creates per-channel quantization parameters.
    pub fn per_channel(scales: Vec<f64>, zero_points: Vec<i32>, dtype: QuantDType) -> Self {
        Self {
            scales,
            zero_points,
            qmin: dtype.qmin(),
            qmax: dtype.qmax(),
            scheme: QuantScheme::AffinePerChannel,
            dtype,
        }
    }
}

/// A quantized tensor holding quantized integer data alongside affine scale/zero-point parameters.
#[derive(Debug, Clone)]
pub struct QuantTensor {
    pub data: Vec<i32>,
    pub shape: Vec<usize>,
    pub params: QParams,
}

impl QuantTensor {
    /// Creates a new QuantTensor from integer data, shape, and quantization parameters.
    pub fn new(data: Vec<i32>, shape: Vec<usize>, params: QParams) -> Self {
        Self { data, shape, params }
    }

    /// Dequantizes the tensor back to high-precision floating point tensor.
    pub fn dequantize(&self) -> Tensor {
        let n = self.data.len();
        let mut float_data = vec![0.0; n];

        match self.params.scheme {
            QuantScheme::AffinePerTensor | QuantScheme::SymmetricPerTensor => {
                let scale = self.params.scales.first().copied().unwrap_or(1.0);
                let zp = self.params.zero_points.first().copied().unwrap_or(0);
                for i in 0..n {
                    float_data[i] = (self.data[i] - zp) as f64 * scale;
                }
            }
            QuantScheme::AffinePerChannel | QuantScheme::SymmetricPerChannel => {
                let num_channels = self.params.scales.len().max(1);
                let channel_size = n / num_channels;
                for ch in 0..num_channels {
                    let scale = self.params.scales.get(ch).copied().unwrap_or(1.0);
                    let zp = self.params.zero_points.get(ch).copied().unwrap_or(0);
                    let start = ch * channel_size;
                    let end = if ch == num_channels - 1 {
                        n
                    } else {
                        (start + channel_size).min(n)
                    };
                    for i in start..end {
                        float_data[i] = (self.data[i] - zp) as f64 * scale;
                    }
                }
            }
            QuantScheme::GroupWise { group_size } => {
                let num_groups = self.params.scales.len().max(1);
                let g_size = group_size.max(1);
                for g in 0..num_groups {
                    let scale = self.params.scales.get(g).copied().unwrap_or(1.0);
                    let zp = self.params.zero_points.get(g).copied().unwrap_or(0);
                    let start = g * g_size;
                    let end = if g == num_groups - 1 {
                        n
                    } else {
                        (start + g_size).min(n)
                    };
                    for i in start..end {
                        if i < n {
                            float_data[i] = (self.data[i] - zp) as f64 * scale;
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&float_data, self.shape.clone())
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Comprehensive error type for quantization failures.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantError {
    EmptyTensor,
    InvalidScale(f64),
    InvalidZeroPoint(i32),
    ShapeMismatch { expected: Vec<usize>, found: Vec<usize> },
    ChannelCountMismatch { expected: usize, found: usize },
    CalibrationError(String),
    UnsupportedDType(QuantDType),
    SparsityError(String),
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantError::EmptyTensor => write!(f, "Cannot quantize empty tensor with zero elements"),
            QuantError::InvalidScale(s) => write!(f, "Invalid quantization scale factor: {}", s),
            QuantError::InvalidZeroPoint(zp) => write!(f, "Invalid zero point offset: {}", zp),
            QuantError::ShapeMismatch { expected, found } => {
                write!(f, "Shape mismatch: expected {:?}, found {:?}", expected, found)
            }
            QuantError::ChannelCountMismatch { expected, found } => {
                write!(f, "Channel count mismatch: expected {}, found {}", expected, found)
            }
            QuantError::CalibrationError(msg) => write!(f, "Calibration error: {}", msg),
            QuantError::UnsupportedDType(dt) => write!(f, "Unsupported quantized data type: {:?}", dt),
            QuantError::SparsityError(msg) => write!(f, "Sparse representation error: {}", msg),
        }
    }
}

impl std::error::Error for QuantError {}

pub type QuantResult<T> = Result<T, QuantError>;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::calibration::*;
    use crate::quantizer::*;
    use crate::prune::*;
    use crate::sparse::*;
    use crate::builder::*;
    use crate::ops::*;
    use crate::utils::*;
    use crate::dtype_map::*;
    use crate::error_analysis::*;
    use crate::bench_quant::*;
    use crate::runtime::*;
    use crate::helper::*;
    use crate::r#impl::*;
    use crate::act_quant::*;
    use crate::block_quant::*;
    use crate::mixed::*;
    use crate::graph_quant::*;
    use crate::fake_quant::*;
    use crate::qlinear::*;
    use crate::qconv::*;
    use crate::qmatmul::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
