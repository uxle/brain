//! # ONNX Model Serializer & Binary Exporter
//!
//! Encodes `OnnxModel` IR back into protobuf binary wire format.
#![allow(missing_docs)]

use crate::core::OnnxResult;
use crate::ir::OnnxModel;
use crate::utils::encode_varint;

fn encode_varint_field(buf: &mut Vec<u8>, field_num: u32, val: u64) {
    buf.extend(encode_varint(((field_num as u64) << 3) | 0));
    buf.extend(encode_varint(val));
}

fn encode_bytes_field(buf: &mut Vec<u8>, field_num: u32, bytes: &[u8]) {
    buf.extend(encode_varint(((field_num as u64) << 3) | 2));
    buf.extend(encode_varint(bytes.len() as u64));
    buf.extend_from_slice(bytes);
}

fn encode_string_field(buf: &mut Vec<u8>, field_num: u32, s: &str) {
    encode_bytes_field(buf, field_num, s.as_bytes());
}

/// Serializes an OnnxModel into raw ONNX binary bytes.
pub fn export_onnx_bytes(model: &OnnxModel) -> OnnxResult<Vec<u8>> {
    let mut buf = Vec::new();

    // Field 1 (ir_version): int64
    encode_varint_field(&mut buf, 1, model.ir_version as u64);

    // Field 2 (producer_name): string
    encode_string_field(&mut buf, 2, &model.producer_name);

    // Field 8 (opset_import): repeated OperatorSetIdProto
    let mut opset_buf = Vec::new();
    encode_string_field(&mut opset_buf, 1, "ai.onnx");
    encode_varint_field(&mut opset_buf, 2, model.opset_version as u64);
    encode_bytes_field(&mut buf, 8, &opset_buf);

    // Field 7 (graph): GraphProto
    let mut graph_buf = Vec::new();
    encode_string_field(&mut graph_buf, 2, &model.graph.name);

    // Graph nodes: Field 1 (node, repeated NodeProto)
    for node in &model.graph.nodes {
        let mut node_buf = Vec::new();
        for inp in &node.inputs {
            encode_string_field(&mut node_buf, 1, inp);
        }
        for out in &node.outputs {
            encode_string_field(&mut node_buf, 2, out);
        }
        encode_string_field(&mut node_buf, 3, &node.name);
        encode_string_field(&mut node_buf, 4, &node.op_type);
        encode_string_field(&mut node_buf, 7, &node.domain);
        encode_bytes_field(&mut graph_buf, 1, &node_buf);
    }

    // Initializers: Field 5 (initializer, repeated TensorProto)
    for (name, val) in &model.graph.values {
        if val.is_initializer {
            if let Some(ref t) = val.tensor_data {
                let mut tensor_buf = Vec::new();
                for &d in &val.shape {
                    encode_varint_field(&mut tensor_buf, 1, d as u64);
                }
                encode_varint_field(&mut tensor_buf, 2, 1); // DataType::Float
                encode_string_field(&mut tensor_buf, 7, name);

                let mut raw_bytes = Vec::with_capacity(t.numel() * 4);
                for &v in t.data() {
                    raw_bytes.extend_from_slice(&(v as f32).to_le_bytes());
                }
                encode_bytes_field(&mut tensor_buf, 9, &raw_bytes);

                encode_bytes_field(&mut graph_buf, 5, &tensor_buf);
            }
        }
    }

    // Inputs: Field 11 (input, repeated ValueInfoProto)
    for in_name in &model.graph.inputs {
        let mut vi_buf = Vec::new();
        encode_string_field(&mut vi_buf, 1, in_name);
        encode_bytes_field(&mut graph_buf, 11, &vi_buf);
    }

    // Outputs: Field 12 (output, repeated ValueInfoProto)
    for out_name in &model.graph.outputs {
        let mut vi_buf = Vec::new();
        encode_string_field(&mut vi_buf, 1, out_name);
        encode_bytes_field(&mut graph_buf, 12, &vi_buf);
    }

    encode_bytes_field(&mut buf, 7, &graph_buf);

    Ok(buf)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
