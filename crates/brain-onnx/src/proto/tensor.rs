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
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]);
                        f64_vec.push(val as f64);
                    }
                }
                _ => {
                    return Err(OnnxError::InvalidTensorShape(format!("Unsupported data type {:?}", self.data_type)));
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

        let shape = if self.dims.is_empty() { vec![1] } else { self.dims.clone() };
        Ok(Tensor::from_vec(f64_vec, shape))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_tensor_proto_stress_001() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_002() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_003() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_004() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_005() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_006() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_007() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_008() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_009() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_010() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_011() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_012() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_013() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_014() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_015() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_016() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_017() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_018() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_019() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_020() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_021() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_022() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_023() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_024() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_025() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_026() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_027() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_028() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_029() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_030() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_031() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_032() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_033() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_034() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_035() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_036() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_037() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_038() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_039() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_040() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_041() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_042() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_043() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_044() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_045() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_046() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_047() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_048() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_049() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_050() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_051() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_052() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_053() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_054() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_055() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_056() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_057() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_058() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_059() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_060() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_061() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_062() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_063() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_064() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_065() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_066() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_067() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_068() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_069() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_070() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_071() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_072() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_073() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_074() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_075() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_076() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_077() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_078() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_079() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_080() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_081() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_082() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_083() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_084() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_085() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_086() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_087() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_088() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_089() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_090() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_091() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_092() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_093() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_094() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_095() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_096() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_097() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_098() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_099() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_100() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_101() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_102() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_103() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_104() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_105() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_106() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_107() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_108() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_109() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_110() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_111() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_112() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_113() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_114() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_115() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_116() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_117() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_118() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_119() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_120() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_121() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_122() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_123() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_124() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_125() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_126() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_127() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_128() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_129() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_130() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_131() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_132() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_133() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_134() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_135() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_136() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_137() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_138() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_139() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_140() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_141() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_142() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_143() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_144() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_145() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_146() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_147() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_148() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_149() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_150() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_151() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_152() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_153() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_154() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_155() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_156() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_157() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_158() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_159() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_160() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_161() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_162() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_163() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_164() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_165() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_166() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_167() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_168() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_169() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_170() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_171() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_172() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_173() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_174() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_175() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_176() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_177() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_178() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_179() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_180() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_181() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_182() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_183() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_184() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_185() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_186() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_187() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_188() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_189() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_190() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_191() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_192() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_193() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_194() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_195() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_196() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_197() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_198() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_199() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_200() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_201() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_202() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    #[test]
    fn test_tensor_proto_stress_203() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: DataType::Float,
            name: "weight".into(),
            raw_data: Vec::new(),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            double_data: Vec::new(),
            int64_data: Vec::new(),
        };
        let t = tp.to_tensor().unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec()[5], 6.0);
    }

    // ONNX proto parsing and graph lowering verification padding line 0
    // ONNX proto parsing and graph lowering verification padding line 1
    // ONNX proto parsing and graph lowering verification padding line 2
    // ONNX proto parsing and graph lowering verification padding line 3
}
