//! # ONNX Wire Protobuf Parser
//!
//! Hand-rolled, zero-dependency protobuf binary wire decoder for ModelProto, GraphProto, NodeProto, and TensorProto.
#![allow(missing_docs)]

pub mod attrs;
pub mod tensor;

pub use attrs::{AttributeProto, AttributeType};
pub use tensor::{DataType, TensorProto};

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

fn parse_string(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).to_string()
}

fn parse_node_proto(bytes: &[u8]) -> OnnxResult<NodeProto> {
    let mut node = NodeProto::default();
    let mut offset = 0;
    while offset < bytes.len() {
        let (tag_wire, next) = decode_varint(bytes, offset)?;
        let wire_type = tag_wire & 0x07;
        let field_num = tag_wire >> 3;
        offset = next;
        match wire_type {
            0 => {
                let (_, next) = decode_varint(bytes, offset)?;
                offset = next;
            }
            2 => {
                let (len, next) = decode_varint(bytes, offset)?;
                let len = len as usize;
                offset = next;
                if offset + len > bytes.len() {
                    return Err(OnnxError::ProtobufDecodeError(
                        "Node length exceeds buffer".into(),
                    ));
                }
                let slice = &bytes[offset..offset + len];
                offset += len;
                match field_num {
                    1 => node.input.push(parse_string(slice)),
                    2 => node.output.push(parse_string(slice)),
                    3 => node.name = parse_string(slice),
                    4 => node.op_type = parse_string(slice),
                    7 => node.domain = parse_string(slice),
                    _ => {}
                }
            }
            _ => break,
        }
    }
    Ok(node)
}

fn parse_tensor_proto(bytes: &[u8]) -> OnnxResult<TensorProto> {
    let mut tensor = TensorProto::default();
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
                match field_num {
                    1 => tensor.dims.push(val as usize),
                    2 => {
                        tensor.data_type = match val {
                            1 => DataType::Float,
                            11 => DataType::Double,
                            7 => DataType::Int64,
                            _ => DataType::Float,
                        }
                    }
                    _ => {}
                }
            }
            2 => {
                let (len, next) = decode_varint(bytes, offset)?;
                let len = len as usize;
                offset = next;
                if offset + len > bytes.len() {
                    return Err(OnnxError::ProtobufDecodeError(
                        "Tensor length exceeds buffer".into(),
                    ));
                }
                let slice = &bytes[offset..offset + len];
                offset += len;
                match field_num {
                    1 => {
                        let mut p_off = 0;
                        while p_off < slice.len() {
                            let (dim, next_p) = decode_varint(slice, p_off)?;
                            tensor.dims.push(dim as usize);
                            p_off = next_p;
                        }
                    }
                    7 => tensor.name = parse_string(slice),
                    9 => tensor.raw_data = slice.to_vec(),
                    _ => {}
                }
            }
            _ => break,
        }
    }
    Ok(tensor)
}

fn parse_value_info_proto(bytes: &[u8]) -> OnnxResult<ValueInfoProto> {
    let mut vi = ValueInfoProto::default();
    let mut offset = 0;
    while offset < bytes.len() {
        let (tag_wire, next) = decode_varint(bytes, offset)?;
        let wire_type = tag_wire & 0x07;
        let field_num = tag_wire >> 3;
        offset = next;
        match wire_type {
            0 => {
                let (_, next) = decode_varint(bytes, offset)?;
                offset = next;
            }
            2 => {
                let (len, next) = decode_varint(bytes, offset)?;
                let len = len as usize;
                offset = next;
                if offset + len > bytes.len() {
                    return Err(OnnxError::ProtobufDecodeError(
                        "ValueInfo length exceeds buffer".into(),
                    ));
                }
                let slice = &bytes[offset..offset + len];
                offset += len;
                if field_num == 1 {
                    vi.name = parse_string(slice);
                }
            }
            _ => break,
        }
    }
    Ok(vi)
}

fn parse_graph_proto(bytes: &[u8]) -> OnnxResult<GraphProto> {
    let mut graph = GraphProto::default();
    let mut offset = 0;
    while offset < bytes.len() {
        let (tag_wire, next) = decode_varint(bytes, offset)?;
        let wire_type = tag_wire & 0x07;
        let field_num = tag_wire >> 3;
        offset = next;
        match wire_type {
            0 => {
                let (_, next) = decode_varint(bytes, offset)?;
                offset = next;
            }
            2 => {
                let (len, next) = decode_varint(bytes, offset)?;
                let len = len as usize;
                offset = next;
                if offset + len > bytes.len() {
                    return Err(OnnxError::ProtobufDecodeError(
                        "Graph length exceeds buffer".into(),
                    ));
                }
                let slice = &bytes[offset..offset + len];
                offset += len;
                match field_num {
                    1 => graph.node.push(parse_node_proto(slice)?),
                    2 => graph.name = parse_string(slice),
                    5 => graph.initializer.push(parse_tensor_proto(slice)?),
                    11 => graph.input.push(parse_value_info_proto(slice)?),
                    12 => graph.output.push(parse_value_info_proto(slice)?),
                    _ => {}
                }
            }
            _ => break,
        }
    }
    Ok(graph)
}

fn parse_opset_proto(bytes: &[u8]) -> OnnxResult<(String, i64)> {
    let mut domain = "ai.onnx".to_string();
    let mut version = 17i64;
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
                if field_num == 2 {
                    version = val as i64;
                }
            }
            2 => {
                let (len, next) = decode_varint(bytes, offset)?;
                let len = len as usize;
                offset = next;
                if offset + len > bytes.len() {
                    return Err(OnnxError::ProtobufDecodeError(
                        "Opset length exceeds buffer".into(),
                    ));
                }
                let slice = &bytes[offset..offset + len];
                offset += len;
                if field_num == 1 {
                    domain = parse_string(slice);
                }
            }
            _ => break,
        }
    }
    Ok((domain, version))
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
                if field_num == 1 {
                    model.ir_version = val as i64;
                } else if field_num == 5 {
                    model.model_version = val as i64;
                }
            }
            2 => {
                let (len, next) = decode_varint(bytes, offset)?;
                let len = len as usize;
                offset = next;
                if offset + len > bytes.len() {
                    return Err(OnnxError::ProtobufDecodeError(
                        "Length exceeds byte buffer".into(),
                    ));
                }
                let slice = &bytes[offset..offset + len];
                offset += len;
                match field_num {
                    2 => model.producer_name = parse_string(slice),
                    3 => model.producer_version = parse_string(slice),
                    4 => model.domain = parse_string(slice),
                    6 => model.doc_string = parse_string(slice),
                    7 => model.graph = Some(parse_graph_proto(slice)?),
                    8 => {
                        let (d, v) = parse_opset_proto(slice)?;
                        model.opset_import = vec![(d, v)];
                    }
                    _ => {}
                }
            }
            _ => {
                break;
            }
        }
    }

    Ok(model)
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
