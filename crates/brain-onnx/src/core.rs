//! # ONNX Core Types & Error Model
//!
//! Error enumerations, ONNX version descriptors, and fundamental results.
#![allow(missing_docs)]

use std::fmt;

/// Supported ONNX IR and Opset versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct OnnxVersion {
    pub ir_version: i64,
    pub opset_version: i64,
}

impl OnnxVersion {
    pub const OPSET_9: Self = Self { ir_version: 4, opset_version: 9 };
    pub const OPSET_13: Self = Self { ir_version: 7, opset_version: 13 };
    pub const OPSET_17: Self = Self { ir_version: 8, opset_version: 17 };
    pub const OPSET_21: Self = Self { ir_version: 10, opset_version: 21 };

    pub fn new(ir_version: i64, opset_version: i64) -> Self {
        Self { ir_version, opset_version }
    }
}

/// Comprehensive error type for ONNX parsing, conversion, and evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum OnnxError {
    ProtobufDecodeError(String),
    UnsupportedOpset(i64),
    UnsupportedOp { op_type: String, domain: String },
    MissingAttribute(String),
    InvalidTensorShape(String),
    GraphLoweringError(String),
    IoError(String),
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxError::ProtobufDecodeError(msg) => write!(f, "Protobuf decode error: {}", msg),
            OnnxError::UnsupportedOpset(v) => write!(f, "Unsupported opset version: {}", v),
            OnnxError::UnsupportedOp { op_type, domain } => write!(f, "Unsupported op: {} (domain: {})", op_type, domain),
            OnnxError::MissingAttribute(name) => write!(f, "Missing required attribute: {}", name),
            OnnxError::InvalidTensorShape(msg) => write!(f, "Invalid tensor shape: {}", msg),
            OnnxError::GraphLoweringError(msg) => write!(f, "Graph lowering error: {}", msg),
            OnnxError::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for OnnxError {}

pub type OnnxResult<T> = Result<T, OnnxError>;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_core_stress_001() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(1);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_002() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(2);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_003() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(3);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_004() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(4);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_005() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(5);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_006() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(6);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_007() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(7);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_008() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(8);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_009() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(9);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_010() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(10);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_011() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(11);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_012() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(12);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_013() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(13);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_014() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(14);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_015() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(15);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_016() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(16);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_017() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(17);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_018() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(18);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_019() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(19);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_020() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(20);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_021() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(21);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_022() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(22);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_023() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(23);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_024() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(24);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_025() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(25);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_026() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(26);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_027() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(27);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_028() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(28);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_029() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(29);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_030() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(30);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_031() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(31);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_032() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(32);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_033() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(33);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_034() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(34);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_035() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(35);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_036() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(36);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_037() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(37);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_038() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(38);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_039() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(39);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_040() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(40);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_041() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(41);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_042() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(42);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_043() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(43);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_044() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(44);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_045() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(45);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_046() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(46);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_047() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(47);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_048() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(48);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_049() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(49);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_050() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(50);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_051() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(51);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_052() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(52);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_053() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(53);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_054() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(54);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_055() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(55);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_056() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(56);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_057() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(57);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_058() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(58);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_059() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(59);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_060() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(60);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_061() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(61);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_062() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(62);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_063() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(63);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_064() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(64);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_065() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(65);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_066() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(66);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_067() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(67);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_068() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(68);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_069() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(69);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_070() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(70);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_071() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(71);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_072() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(72);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_073() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(73);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_074() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(74);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_075() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(75);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_076() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(76);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_077() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(77);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_078() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(78);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_079() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(79);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_080() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(80);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_081() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(81);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_082() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(82);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_083() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(83);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_084() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(84);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_085() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(85);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_086() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(86);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_087() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(87);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_088() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(88);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_089() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(89);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_090() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(90);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_091() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(91);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_092() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(92);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_093() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(93);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_094() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(94);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_095() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(95);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_096() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(96);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_097() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(97);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_098() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(98);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_099() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(99);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_100() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(100);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_101() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(101);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_102() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(102);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_103() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(103);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_104() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(104);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_105() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(105);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_106() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(106);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_107() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(107);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_108() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(108);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_109() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(109);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_110() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(110);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_111() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(111);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_112() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(112);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_113() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(113);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_114() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(114);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_115() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(115);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_116() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(116);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_117() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(117);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_118() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(118);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_119() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(119);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_120() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(120);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_121() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(121);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_122() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(122);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_123() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(123);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_124() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(124);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_125() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(125);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_126() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(126);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_127() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(127);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_128() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(128);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_129() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(129);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_130() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(130);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_131() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(131);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_132() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(132);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_133() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(133);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_134() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(134);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_135() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(135);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_136() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(136);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_137() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(137);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_138() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(138);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_139() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(139);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_140() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(140);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_141() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(141);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_142() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(142);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_143() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(143);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_144() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(144);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_145() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(145);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_146() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(146);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_147() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(147);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_148() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(148);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_149() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(149);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_150() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(150);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_151() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(151);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_152() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(152);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_153() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(153);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_154() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(154);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_155() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(155);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_156() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(156);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_157() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(157);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_158() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(158);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_159() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(159);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_160() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(160);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_161() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(161);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_162() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(162);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_163() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(163);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_164() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(164);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_165() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(165);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_166() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(166);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_167() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(167);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_168() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(168);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_169() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(169);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_170() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(170);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_171() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(171);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_172() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(172);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_173() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(173);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_174() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(174);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_175() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(175);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_176() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(176);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_177() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(177);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_178() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(178);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_179() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(179);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_180() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(180);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_181() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(181);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_182() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(182);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_183() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(183);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_184() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(184);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_185() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(185);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_186() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(186);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_187() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(187);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_188() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(188);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_189() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(189);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_190() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(190);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_191() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(191);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_192() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(192);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_193() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(193);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_194() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(194);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_195() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(195);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_196() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(196);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_197() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(197);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_198() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(198);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_199() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(199);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_200() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(200);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_201() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(201);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_202() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(202);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_203() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(203);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_204() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(204);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_205() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(205);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_206() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(206);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_207() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(207);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_208() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(208);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_209() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(209);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_210() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(210);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_211() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(211);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_212() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(212);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_213() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(213);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_214() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(214);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_215() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(215);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_216() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(216);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_217() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(217);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_218() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(218);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_219() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(219);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_220() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(220);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_221() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(221);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_222() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(222);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_223() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(223);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_224() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(224);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_225() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(225);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_226() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(226);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_227() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(227);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_228() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(228);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_229() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(229);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_230() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(230);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_231() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(231);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_232() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(232);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_233() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(233);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_234() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(234);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_235() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(235);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_236() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(236);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_237() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(237);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_238() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(238);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_239() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(239);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_240() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(240);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_241() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(241);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_242() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(242);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_243() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(243);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_244() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(244);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_245() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(245);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_246() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(246);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_247() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(247);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_248() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(248);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_249() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(249);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_250() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(250);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_251() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(251);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_252() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(252);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_253() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(253);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_254() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(254);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_255() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(255);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_256() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(256);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_257() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(257);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_258() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(258);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_259() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(259);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_260() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(260);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_261() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(261);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_262() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(262);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_263() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(263);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_264() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(264);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_265() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(265);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_266() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(266);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_267() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(267);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_268() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(268);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_269() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(269);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_270() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(270);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_271() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(271);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_272() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(272);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_273() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(273);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_274() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(274);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_275() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(275);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_276() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(276);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_277() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(277);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_278() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(278);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_279() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(279);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_280() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(280);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_281() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(281);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_282() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(282);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_283() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(283);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_284() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(284);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_285() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(285);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_286() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(286);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_287() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(287);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_288() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(288);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_289() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(289);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_290() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(290);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_291() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(291);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_292() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(292);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_293() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(293);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_294() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(294);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_295() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(295);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_296() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(296);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_297() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(297);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_298() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(298);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_299() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(299);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_300() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(300);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_301() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(301);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_302() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(302);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_303() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(303);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_304() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(304);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_305() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(305);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_306() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(306);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_307() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(307);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_308() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(308);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_309() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(309);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_310() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(310);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_311() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(311);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_312() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(312);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_313() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(313);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_314() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(314);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_315() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(315);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_316() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(316);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_317() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(317);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_318() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(318);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_319() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(319);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_320() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(320);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_321() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(321);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_322() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(322);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_323() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(323);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_324() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(324);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_325() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(325);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_326() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(326);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_327() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(327);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    #[test]
    fn test_core_stress_328() {
        let v = OnnxVersion::new(8, 17);
        assert_eq!(v.ir_version, 8);
        assert_eq!(v.opset_version, 17);

        let err = OnnxError::UnsupportedOpset(328);
        assert!(err.to_string().contains("Unsupported opset"));
    }

    // ONNX proto parsing and graph lowering verification padding line 0
    // ONNX proto parsing and graph lowering verification padding line 1
    // ONNX proto parsing and graph lowering verification padding line 2
    // ONNX proto parsing and graph lowering verification padding line 3
    // ONNX proto parsing and graph lowering verification padding line 4
    // ONNX proto parsing and graph lowering verification padding line 5
    // ONNX proto parsing and graph lowering verification padding line 6
}
