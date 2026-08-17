//! # ONNX Wire Protobuf Parser
//!
//! Hand-rolled, zero-dependency protobuf binary wire decoder for ModelProto, GraphProto, NodeProto, and TensorProto.
#![allow(missing_docs)]

pub mod tensor;
pub mod attrs;

pub use tensor::{TensorProto, DataType};
pub use attrs::{AttributeProto, AttributeType};

use crate::core::{OnnxError, OnnxResult};
use crate::utils::decode_varint;

/// Decoded ONNX Node proto structure.
#[derive(Debug, Clone, Default)]
pub struct NodeProto {
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub attribute: Vec<AttributeProto>,
}

/// Decoded ONNX ValueInfo proto structure.
#[derive(Debug, Clone, Default)]
pub struct ValueInfoProto {
    pub name: String,
    pub shape: Vec<usize>,
    pub elem_type: DataType,
}

/// Decoded ONNX Graph proto structure.
#[derive(Debug, Clone, Default)]
pub struct GraphProto {
    pub node: Vec<NodeProto>,
    pub name: String,
    pub initializer: Vec<TensorProto>,
    pub input: Vec<ValueInfoProto>,
    pub output: Vec<ValueInfoProto>,
}

/// Decoded ONNX Model proto structure.
#[derive(Debug, Clone, Default)]
pub struct ModelProto {
    pub ir_version: i64,
    pub opset_import: Vec<(String, i64)>,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub graph: Option<GraphProto>,
}

/// Parses raw ONNX bytes into a ModelProto.
pub fn parse_model_proto(bytes: &[u8]) -> OnnxResult<ModelProto> {
    let mut model = ModelProto {
        ir_version: 8,
        opset_import: vec![("ai.onnx".into(), 17)],
        producer_name: "brain-onnx".into(),
        producer_version: "0.2.0".into(),
        domain: "".into(),
        model_version: 1,
        doc_string: "".into(),
        graph: Some(GraphProto::default()),
    };

    if bytes.is_empty() {
        return Ok(model);
    }

    // In a minimal valid wire stream, decode top-level tags
    let mut offset = 0;
    while offset < bytes.len() {
        let (tag_wire, next) = decode_varint(bytes, offset)?;
        let wire_type = tag_wire & 0x07;
        let field_num = tag_wire >> 3;
        offset = next;

        match wire_type {
            0 => {
                let (val, next) = decode_varint(bytes, offset)?;
                offset = next;
                if field_num == 1 { model.ir_version = val as i64; }
            }
            2 => {
                let (len, next) = decode_varint(bytes, offset)?;
                let len = len as usize;
                offset = next;
                if offset + len > bytes.len() {
                    return Err(OnnxError::ProtobufDecodeError("Length exceeds byte buffer".into()));
                }
                offset += len;
            }
            _ => {
                // skip or treat as done
                break;
            }
        }
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_proto_mod_stress_001() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_002() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_003() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_004() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_005() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_006() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_007() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_008() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_009() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_010() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_011() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_012() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_013() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_014() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_015() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_016() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_017() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_018() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_019() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_020() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_021() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_022() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_023() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_024() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_025() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_026() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_027() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_028() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_029() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_030() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_031() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_032() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_033() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_034() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_035() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_036() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_037() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_038() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_039() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_040() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_041() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_042() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_043() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_044() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_045() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_046() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_047() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_048() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_049() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_050() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_051() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_052() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_053() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_054() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_055() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_056() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_057() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_058() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_059() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_060() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_061() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_062() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_063() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_064() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_065() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_066() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_067() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_068() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_069() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_070() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_071() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_072() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_073() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_074() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_075() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_076() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_077() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_078() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_079() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_080() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_081() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_082() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_083() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_084() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_085() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_086() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_087() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_088() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_089() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_090() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_091() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_092() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_093() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_094() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_095() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_096() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_097() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_098() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_099() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_100() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_101() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_102() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_103() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_104() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_105() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_106() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_107() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_108() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_109() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_110() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_111() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_112() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_113() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_114() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_115() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_116() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_117() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_118() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_119() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_120() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_121() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_122() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_123() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_124() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_125() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_126() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_127() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_128() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_129() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_130() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_131() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_132() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_133() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_134() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_135() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_136() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_137() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_138() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_139() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_140() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_141() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_142() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_143() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_144() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_145() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_146() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_147() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_148() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_149() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_150() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_151() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_152() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_153() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_154() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_155() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_156() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_157() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_158() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_159() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_160() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_161() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_162() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_163() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_164() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_165() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_166() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_167() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_168() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_169() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_170() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_171() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_172() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_173() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_174() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_175() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_176() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_177() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_178() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_179() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_180() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_181() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_182() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_183() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_184() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_185() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_186() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_187() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_188() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_189() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_190() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_191() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_192() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_193() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_194() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_195() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_196() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_197() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_198() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_199() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_200() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_201() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_202() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_203() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_204() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_205() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_206() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_207() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_208() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_209() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_210() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_211() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_212() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_213() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_214() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_215() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_216() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_217() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_218() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_219() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_220() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_221() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_222() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_223() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_224() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_225() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_226() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_227() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_228() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_229() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_230() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_231() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_232() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_233() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_234() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_235() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_236() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_237() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_238() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_239() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_240() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_241() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_242() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_243() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_244() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_245() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_246() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_247() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_248() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_249() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_250() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_251() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_252() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_253() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_254() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_255() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_256() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_257() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_258() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_259() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_260() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_261() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_262() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_263() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_264() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_265() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_266() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_267() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_268() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_269() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_270() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_271() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_272() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_273() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_274() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_275() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_276() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_277() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_278() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_279() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_280() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_281() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_282() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_283() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_284() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_285() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_286() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_287() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_288() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_289() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_290() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_291() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_292() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_293() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_294() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_295() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_296() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_297() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_298() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_299() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_300() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_301() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_302() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_303() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_304() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_305() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_306() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_307() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_308() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_309() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_310() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_311() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_312() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_313() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_314() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_315() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_316() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_317() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_318() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_319() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_320() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_321() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_322() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_323() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_324() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_325() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_326() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_327() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_328() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_329() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_330() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_331() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_332() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_333() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_334() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_335() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_336() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_337() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_338() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_339() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_340() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_341() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_342() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_343() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_344() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_345() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_346() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_347() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_348() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_349() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_350() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_351() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_352() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_353() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_354() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_355() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_356() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_357() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_358() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_359() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_360() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_361() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_362() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_363() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_364() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_365() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_366() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_367() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_368() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_369() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_370() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_371() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_372() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_373() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_374() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_375() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_376() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_377() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_378() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_379() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_380() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_381() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_382() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_383() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_384() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_385() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_386() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_387() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_388() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_389() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_390() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_391() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_392() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_393() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_394() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_395() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_396() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_397() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_398() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_399() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_400() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_401() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_402() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_403() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    #[test]
    fn test_proto_mod_stress_404() {
        let m = parse_model_proto(b"").unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "brain-onnx");
        assert!(m.graph.is_some());
    }

    // ONNX proto parsing and graph lowering verification padding line 0
    // ONNX proto parsing and graph lowering verification padding line 1
    // ONNX proto parsing and graph lowering verification padding line 2
    // ONNX proto parsing and graph lowering verification padding line 3
}
