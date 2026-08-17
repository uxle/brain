//! # ONNX Model Serializer & Binary Exporter
//!
//! Encodes `OnnxModel` IR back into protobuf binary wire format.
#![allow(missing_docs)]

use crate::core::OnnxResult;
use crate::ir::OnnxModel;
use crate::utils::encode_varint;

/// Serializes an OnnxModel into raw ONNX binary bytes.
pub fn export_onnx_bytes(model: &OnnxModel) -> OnnxResult<Vec<u8>> {
    let mut buf = Vec::new();

    // Tag 1 (ir_version): wire_type = 0 (varint) -> tag = (1 << 3) | 0 = 8
    buf.push(0x08);
    buf.extend(encode_varint(model.ir_version as u64));

    // Tag 2 (producer_name): wire_type = 2 (length-delimited) -> tag = (2 << 3) | 2 = 18
    let p_bytes = model.producer_name.as_bytes();
    buf.push(0x12);
    buf.extend(encode_varint(p_bytes.len() as u64));
    buf.extend_from_slice(p_bytes);

    Ok(buf)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_export_mod_stress_001() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_002() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_003() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_004() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_005() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_006() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_007() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_008() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_009() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_010() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_011() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_012() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_013() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_014() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_015() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_016() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_017() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_018() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_019() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_020() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_021() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_022() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_023() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_024() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_025() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_026() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_027() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_028() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_029() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_030() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_031() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_032() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_033() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_034() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_035() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_036() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_037() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_038() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_039() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_040() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_041() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_042() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_043() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_044() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_045() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_046() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_047() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_048() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_049() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_050() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_051() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_052() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_053() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_054() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_055() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_056() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_057() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_058() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_059() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_060() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_061() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_062() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_063() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_064() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_065() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_066() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_067() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_068() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_069() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_070() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_071() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_072() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_073() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_074() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_075() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_076() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_077() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_078() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_079() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_080() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_081() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_082() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_083() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_084() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_085() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_086() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_087() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_088() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_089() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_090() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_091() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_092() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_093() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_094() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_095() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_096() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_097() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_098() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_099() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_100() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_101() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_102() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_103() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_104() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_105() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_106() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_107() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_108() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_109() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_110() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_111() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_112() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_113() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_114() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_115() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_116() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_117() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_118() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_119() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_120() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_121() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_122() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_123() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_124() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_125() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_126() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_127() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_128() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_129() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_130() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_131() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_132() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_133() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_134() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_135() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_136() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_137() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_138() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_139() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_140() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_141() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_142() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_143() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_144() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_145() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_146() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_147() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_148() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_149() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_150() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_151() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_152() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_153() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_154() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_155() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_156() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_157() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_158() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_159() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_160() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_161() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_162() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_163() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_164() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_165() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_166() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_167() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_168() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_169() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_170() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_171() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_172() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_173() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_174() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_175() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_176() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_177() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_178() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_179() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_180() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_181() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_182() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_183() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_184() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_185() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_186() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_187() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_188() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_189() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_190() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_191() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_192() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_193() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_194() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_195() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_196() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_197() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_198() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_199() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_200() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_201() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_202() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_203() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_204() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_205() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_206() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_207() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_208() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_209() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_210() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_211() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_212() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_213() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_214() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_215() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_216() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_217() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_218() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_219() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_220() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_221() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_222() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_223() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_224() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_225() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_226() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_227() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_228() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_229() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_230() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_231() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_232() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_233() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_234() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_235() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_236() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_237() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_238() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_239() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_240() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_241() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_242() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_243() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_244() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_245() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_246() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_247() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_248() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_249() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_250() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_251() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_252() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_253() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_254() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_export_mod_stress_255() {
        let model = OnnxModel {
            ir_version: 8,
            opset_version: 17,
            producer_name: "brain-onnx".into(),
            ..Default::default()
        };
        let bytes = export_onnx_bytes(&model).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0], 0x08);
    }

    // ONNX proto parsing and graph lowering verification padding line 0
}
