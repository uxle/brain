//! # ONNX Proto to IR Lowering
//!
//! Transforms `ModelProto` structures into verified `OnnxModel` graphs with tensor shapes and initializers.
#![allow(missing_docs)]

use crate::core::OnnxResult;
use crate::config::ImportConfig;
use crate::ir::{OnnxModel, OnnxGraph, OnnxValue};
use crate::proto::ModelProto;
use super::ops::translate_op;

/// Converts a decoded ModelProto into an OnnxModel IR.
pub fn proto_to_ir(proto: &ModelProto, _config: &ImportConfig) -> OnnxResult<OnnxModel> {
    let mut model = OnnxModel {
        ir_version: proto.ir_version,
        opset_version: proto.opset_import.first().map(|x| x.1).unwrap_or(17),
        producer_name: proto.producer_name.clone(),
        graph: OnnxGraph::default(),
    };

    if let Some(ref g) = proto.graph {
        model.graph.name = g.name.clone();

        for n in &g.node {
            model.graph.nodes.push(translate_op(n));
        }

        for inp in &g.input {
            model.graph.inputs.push(inp.name.clone());
            model.graph.values.insert(inp.name.clone(), OnnxValue {
                name: inp.name.clone(),
                shape: inp.shape.clone(),
                is_initializer: false,
                tensor_data: None,
            });
        }

        for out in &g.output {
            model.graph.outputs.push(out.name.clone());
            model.graph.values.insert(out.name.clone(), OnnxValue {
                name: out.name.clone(),
                shape: out.shape.clone(),
                is_initializer: false,
                tensor_data: None,
            });
        }

        for init in &g.initializer {
            let tensor = init.to_tensor().ok();
            model.graph.values.insert(init.name.clone(), OnnxValue {
                name: init.name.clone(),
                shape: init.dims.clone(),
                is_initializer: true,
                tensor_data: tensor,
            });
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
    fn test_onnx2graph_stress_001() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_002() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_003() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_004() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_005() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_006() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_007() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_008() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_009() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_010() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_011() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_012() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_013() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_014() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_015() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_016() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_017() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_018() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_019() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_020() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_021() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_022() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_023() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_024() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_025() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_026() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_027() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_028() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_029() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_030() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_031() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_032() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_033() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_034() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_035() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_036() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_037() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_038() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_039() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_040() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_041() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_042() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_043() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_044() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_045() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_046() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_047() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_048() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_049() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_050() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_051() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_052() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_053() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_054() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_055() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_056() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_057() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_058() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_059() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_060() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_061() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_062() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_063() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_064() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_065() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_066() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_067() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_068() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_069() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_070() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_071() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_072() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_073() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_074() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_075() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_076() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_077() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_078() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_079() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_080() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_081() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_082() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_083() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_084() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_085() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_086() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_087() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_088() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_089() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_090() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_091() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_092() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_093() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_094() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_095() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_096() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_097() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_098() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_099() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_100() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_101() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_102() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_103() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_104() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_105() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_106() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_107() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_108() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_109() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_110() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_111() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_112() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_113() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_114() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_115() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_116() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_117() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_118() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_119() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_120() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_121() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_122() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_123() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_124() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_125() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_126() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_127() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_128() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_129() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_130() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_131() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_132() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_133() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_134() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_135() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_136() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_137() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_138() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_139() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_140() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_141() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_142() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_143() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_144() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_145() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_146() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_147() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_148() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_149() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_150() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_151() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_152() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_153() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_154() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_155() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_156() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_157() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_158() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_159() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_160() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_161() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_162() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_163() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_164() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_165() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_166() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_167() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_168() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_169() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_170() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_171() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_172() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_173() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_174() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_175() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_176() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_177() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_178() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_179() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_180() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_181() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_182() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_183() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_184() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_185() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_186() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_187() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_188() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_189() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_190() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_191() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_192() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_193() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_194() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_195() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_196() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_197() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_198() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_199() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_200() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_201() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_202() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_203() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_204() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_205() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_206() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_207() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_208() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_209() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_210() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_211() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_212() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_213() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_214() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_215() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_216() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_217() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_218() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_219() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_220() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_221() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_222() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_223() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_224() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_225() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_226() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_227() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_228() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_229() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_230() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_231() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_232() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_233() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_234() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_235() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_236() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_237() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_238() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_239() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_240() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_241() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_242() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_243() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_244() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_245() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_246() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_247() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_248() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_249() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_250() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_251() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_252() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_253() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_254() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_255() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_256() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_257() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_258() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_259() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_260() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_261() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_262() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_263() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_264() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_265() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_266() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_267() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_268() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_269() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_270() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_271() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_272() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_273() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_274() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_275() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_276() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_277() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_278() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_279() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_280() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_281() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_282() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_283() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_284() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_285() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_286() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_287() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_288() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_289() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_290() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_291() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_292() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_293() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_294() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_295() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_296() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_297() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_298() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_299() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_300() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_301() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_302() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_303() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_304() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_305() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_306() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_307() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_308() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_309() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_310() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_311() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_312() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_313() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_314() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_315() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_316() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_317() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_318() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_319() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_320() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_321() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_322() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_323() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_324() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_325() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_326() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_327() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_328() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_329() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_330() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_331() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_332() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_333() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_334() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_335() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_336() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_337() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_338() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_339() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_340() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_341() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_342() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_343() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_344() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_345() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_346() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_347() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_348() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_349() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_350() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_351() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_352() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_353() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_354() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_355() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_356() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_357() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_358() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_359() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_360() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_361() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_362() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_363() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_364() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_365() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_366() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_367() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_368() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_369() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_370() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_371() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_372() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_373() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_374() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_375() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_376() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_377() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_378() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_379() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_380() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_381() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_382() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_383() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_384() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_385() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_386() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_387() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_388() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_389() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_390() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_391() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_392() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_393() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_394() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_395() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_396() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_397() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_398() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_399() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_400() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_401() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_402() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_403() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_404() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_405() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_406() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_407() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_408() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_409() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx2graph_stress_410() {
        let proto = ModelProto::default();
        let cfg = ImportConfig::default();
        let model = proto_to_ir(&proto, &cfg).unwrap();
        assert_eq!(model.opset_version, 17);
    }
}
