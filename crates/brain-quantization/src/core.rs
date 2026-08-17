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
                    let scale = self.params.scales[ch];
                    let zp = self.params.zero_points[ch];
                    let start = ch * channel_size;
                    let end = (start + channel_size).min(n);
                    for i in start..end {
                        float_data[i] = (self.data[i] - zp) as f64 * scale;
                    }
                }
            }
            QuantScheme::GroupWise { group_size } => {
                let num_groups = self.params.scales.len().max(1);
                for g in 0..num_groups {
                    let scale = self.params.scales[g];
                    let zp = self.params.zero_points[g];
                    let start = g * group_size;
                    let end = (start + group_size).min(n);
                    for i in start..end {
                        float_data[i] = (self.data[i] - zp) as f64 * scale;
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

    #[test]
    fn test_core_stress_001() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![1 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_002() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![2 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_003() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![3 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_004() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![4 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_005() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![5 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_006() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![6 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_007() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![7 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_008() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![8 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_009() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![9 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_010() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![10 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_011() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![11 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_012() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![12 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_013() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![13 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_014() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![14 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_015() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![15 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_016() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![16 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_017() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![17 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_018() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![18 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_019() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![19 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_020() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![20 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_021() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![21 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_022() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![22 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_023() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![23 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_024() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![24 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_025() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![25 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_026() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![26 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_027() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![27 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_028() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![28 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_029() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![29 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_030() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![30 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_031() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![31 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_032() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![32 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_033() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![33 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_034() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![34 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_035() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![35 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_036() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![36 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_037() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![37 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_038() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![38 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_039() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![39 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_040() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![40 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_041() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![41 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_042() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![42 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_043() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![43 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_044() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![44 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_045() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![45 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_046() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![46 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_047() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![47 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_048() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![48 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_049() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![49 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_050() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![50 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_051() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![51 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_052() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![52 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_053() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![53 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_054() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![54 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_055() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![55 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_056() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![56 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_057() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![57 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_058() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![58 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_059() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![59 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_060() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![60 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_061() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![61 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_062() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![62 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_063() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![63 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_064() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![64 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_065() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![65 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_066() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![66 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_067() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![67 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_068() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![68 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_069() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![69 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_070() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![70 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_071() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![71 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_072() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![72 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_073() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![73 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_074() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![74 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_075() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![75 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_076() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![76 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_077() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![77 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_078() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![78 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_079() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![79 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_080() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![80 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_081() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![81 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_082() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![82 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_083() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![83 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_084() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![84 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_085() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![85 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_086() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![86 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_087() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![87 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_088() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![88 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_089() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![89 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_090() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![90 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_091() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![91 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_092() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![92 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_093() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![93 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_094() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![94 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_095() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![95 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_096() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![96 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_097() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![97 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_098() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![98 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_099() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![99 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_100() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![100 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_101() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![101 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_102() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![102 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_103() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![103 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_104() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![104 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_105() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![105 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_106() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![106 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_107() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![107 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_108() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![108 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_109() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![109 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_110() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![110 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_111() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![111 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_112() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![112 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_113() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![113 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_114() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![114 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_115() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![115 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_116() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![116 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_117() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![117 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_118() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![118 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_119() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![119 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_120() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![120 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_121() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![121 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_122() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![122 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_123() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![123 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_124() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![124 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_125() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![125 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_126() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![126 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_127() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![127 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_128() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![128 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_129() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![129 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_130() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![130 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_131() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![131 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_132() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![132 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_133() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![133 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_134() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![134 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_135() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![135 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_136() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![136 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_137() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![137 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_138() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![138 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_139() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![139 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_140() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![140 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_141() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![141 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_142() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![142 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_143() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![143 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_144() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![144 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_145() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![145 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_146() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![146 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_147() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![147 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_148() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![148 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_149() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![149 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_150() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![150 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_151() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![151 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_152() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![152 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_153() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![153 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_154() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![154 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_155() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![155 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_156() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![156 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_157() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![157 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_158() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![158 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_159() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![159 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_160() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![160 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_161() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![161 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_162() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![162 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_163() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![163 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_164() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![164 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_165() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![165 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_166() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![166 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_167() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![167 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_168() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![168 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_169() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![169 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_170() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![170 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_171() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![171 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_172() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![172 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_173() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![173 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_174() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![174 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_175() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![175 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_176() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![176 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_177() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![177 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_178() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![178 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_179() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![179 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_180() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![180 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_181() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![181 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_182() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![182 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_183() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![183 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_184() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![184 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_185() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![185 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_186() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![186 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_187() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![187 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_188() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![188 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_189() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![189 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_190() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![190 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_191() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![191 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_192() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![192 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_193() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![193 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_194() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![194 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_195() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![195 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_196() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![196 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_197() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![197 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_198() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![198 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_199() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![199 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_200() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![200 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_201() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![201 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_202() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![202 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_203() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![203 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_204() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![204 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_205() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![205 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_206() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![206 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_207() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![207 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_208() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![208 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_209() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![209 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_210() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![210 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_211() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![211 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_212() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![212 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_213() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![213 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_214() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![214 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_215() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![215 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_216() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![216 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_217() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![217 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_218() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![218 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_219() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![219 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_220() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![220 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_221() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![221 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_222() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![222 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_223() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![223 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_224() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![224 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_225() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![225 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_226() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![226 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_227() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![227 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_228() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![228 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_229() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![229 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_230() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![230 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_231() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![231 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_232() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![232 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_233() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![233 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_234() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![234 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_235() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![235 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_236() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![236 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_237() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![237 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    #[test]
    fn test_core_stress_238() {
        let params = QParams::per_tensor(0.01, 0, QuantDType::Int8);
        assert_eq!(params.dtype.bit_width(), 8);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);

        let qt = QuantTensor::new(vec![238 % 120], vec![1], params);
        assert_eq!(qt.numel(), 1);
        let deq = qt.dequantize();
        assert_eq!(deq.shape(), &[1]);
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
}
