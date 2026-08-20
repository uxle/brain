//! # ONNX Operator Importers & Attribute Translators
//!
//! Maps ONNX primitive operators and attributes into canonical internal operator descriptors.
#![allow(missing_docs)]

use crate::ir::OnnxNode;
use crate::proto::NodeProto;
use std::collections::HashMap;

/// Translates a NodeProto into a canonical OnnxNode IR.
pub fn translate_op(proto: &NodeProto) -> OnnxNode {
    let mut attributes = HashMap::new();
    for attr in &proto.attribute {
        let val_str = match attr.attr_type {
            crate::proto::AttributeType::Int => attr.i.to_string(),
            crate::proto::AttributeType::Float => attr.f.to_string(),
            crate::proto::AttributeType::String => attr.s.clone(),
            _ => format!("{:?}", attr.attr_type),
        };
        attributes.insert(attr.name.clone(), val_str);
    }

    OnnxNode {
        name: if proto.name.is_empty() {
            format!(
                "{}_{}",
                proto.op_type,
                proto.output.first().cloned().unwrap_or_default()
            )
        } else {
            proto.name.clone()
        },
        op_type: proto.op_type.clone(),
        domain: proto.domain.clone(),
        inputs: proto.input.clone(),
        outputs: proto.output.clone(),
        attributes,
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
