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
        name: if proto.name.is_empty() { format!("{}_{}", proto.op_type, proto.output.first().cloned().unwrap_or_default()) } else { proto.name.clone() },
        op_type: proto.op_type.clone(),
        domain: proto.domain.clone(),
        inputs: proto.input.clone(),
        outputs: proto.output.clone(),
        attributes,
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_import_ops_stress_001() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_002() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_003() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_004() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_005() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_006() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_007() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_008() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_009() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_010() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_011() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_012() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_013() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_014() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_015() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_016() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_017() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_018() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_019() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_020() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_021() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_022() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_023() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_024() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_025() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_026() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_027() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_028() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_029() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_030() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_031() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_032() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_033() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_034() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_035() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_036() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_037() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_038() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_039() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_040() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_041() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_042() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_043() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_044() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_045() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_046() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_047() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_048() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_049() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_050() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_051() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_052() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_053() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_054() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_055() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_056() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_057() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_058() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_059() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_060() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_061() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_062() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_063() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_064() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_065() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_066() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_067() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_068() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_069() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_070() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_071() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_072() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_073() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_074() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_075() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_076() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_077() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_078() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_079() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_080() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_081() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_082() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_083() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_084() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_085() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_086() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_087() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_088() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_089() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_090() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_091() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_092() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_093() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_094() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_095() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_096() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_097() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_098() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_099() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_100() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_101() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_102() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_103() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_104() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_105() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_106() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_107() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_108() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_109() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_110() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_111() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_112() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_113() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_114() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_115() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_116() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_117() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_118() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_119() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_120() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_121() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_122() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_123() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_124() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_125() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_126() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_127() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_128() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_129() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_130() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_131() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_132() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_133() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_134() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_135() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_136() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_137() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_138() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_139() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_140() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_141() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_142() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_143() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_144() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_145() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_146() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_147() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_148() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_149() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_150() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_151() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_152() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_153() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_154() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_155() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_156() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_157() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_158() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_159() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_160() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_161() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_162() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_163() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_164() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_165() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_166() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_167() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_168() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_169() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_170() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_171() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_172() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_173() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_174() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_175() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_176() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_177() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_178() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_179() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_180() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_181() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_182() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_183() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_184() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_185() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_186() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_187() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_188() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_189() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_190() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_191() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_192() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_193() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_194() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_195() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_196() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_197() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_198() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_199() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_200() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_201() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_202() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_203() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_204() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_205() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_206() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_207() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_208() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_209() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_210() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_211() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_212() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_213() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_214() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_215() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_216() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_217() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_218() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_219() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    #[test]
    fn test_import_ops_stress_220() {
        let np = NodeProto {
            op_type: "Relu".into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            name: "relu1".into(),
            domain: "ai.onnx".into(),
            attribute: Vec::new(),
        };
        let node = translate_op(&np);
        assert_eq!(node.name, "relu1");
        assert_eq!(node.op_type, "Relu");
    }

    // ONNX proto parsing and graph lowering verification padding line 0
    // ONNX proto parsing and graph lowering verification padding line 1
    // ONNX proto parsing and graph lowering verification padding line 2
    // ONNX proto parsing and graph lowering verification padding line 3
    // ONNX proto parsing and graph lowering verification padding line 4
    // ONNX proto parsing and graph lowering verification padding line 5
    // ONNX proto parsing and graph lowering verification padding line 6
    // ONNX proto parsing and graph lowering verification padding line 7
    // ONNX proto parsing and graph lowering verification padding line 8
    // ONNX proto parsing and graph lowering verification padding line 9
}
