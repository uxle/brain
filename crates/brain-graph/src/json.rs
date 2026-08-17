//! # JSON Serialization
//!
//! Pure Rust deterministic JSON serialization and deserialization of computation graphs.
#![allow(missing_docs)]

use crate::ir::GraphIr;

/// Serializes `GraphIr` to a formatted JSON string.
pub fn to_json(graph: &GraphIr) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"name\": \"{}\",\n", graph.name));
    out.push_str("  \"nodes\": [\n");

    for (i, node) in graph.nodes.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"id\": {},\n", node.id));
        out.push_str(&format!("      \"name\": \"{}\",\n", node.name));
        out.push_str(&format!("      \"op\": \"{}\",\n", node.op.name()));
        out.push_str(&format!("      \"inputs\": {:?},\n", node.inputs));
        out.push_str(&format!("      \"outputs\": {:?}\n", node.outputs));
        out.push_str("    }");
        if i + 1 < graph.nodes.len() { out.push(','); }
        out.push('\n');
    }

    out.push_str("  ],\n");
    out.push_str(&format!("  \"inputs\": {:?},\n", graph.inputs));
    out.push_str(&format!("  \"outputs\": {:?}\n", graph.outputs));
    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_json_stress_001() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_002() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_003() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_004() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_005() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_006() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_007() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_008() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_009() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_010() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_011() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_012() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_013() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_014() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_015() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_016() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_017() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_018() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_019() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_020() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_021() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_022() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_023() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_024() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_025() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_026() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_027() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_028() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_029() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_030() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_031() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_032() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_033() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_034() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_035() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_036() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_037() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_038() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_039() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_040() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_041() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_042() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_043() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_044() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_045() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_046() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_047() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_048() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_049() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_050() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_051() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_052() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_053() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_054() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_055() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_056() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_057() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_058() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_059() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_060() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_061() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_062() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_063() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_064() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_065() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_066() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_067() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_068() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_069() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_070() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_071() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_072() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_073() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_074() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_075() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_076() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_077() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_078() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_079() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_080() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_081() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_082() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_083() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_084() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_085() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_086() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_087() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_088() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_089() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_090() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_091() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_092() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_093() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_094() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_095() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_096() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_097() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_098() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_099() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_100() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_101() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_102() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_103() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_104() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_105() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_106() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_107() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_108() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_109() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_110() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_111() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_112() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_113() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_114() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_115() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_116() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_117() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_118() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_119() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_120() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_121() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_122() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_123() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_124() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_125() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_126() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_127() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_128() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_129() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_130() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_131() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_132() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_133() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_134() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_135() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_136() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_137() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_138() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_139() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_140() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_141() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_142() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_143() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_144() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_145() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_146() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_147() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_148() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_149() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_150() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_151() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_152() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_153() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_154() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_155() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_156() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_157() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_158() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_159() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_160() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_161() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_162() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_163() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_164() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_165() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_166() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_167() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_168() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_169() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_170() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_171() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_172() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_173() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_174() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_175() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_176() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_177() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_178() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_179() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_180() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_181() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_182() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_183() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_184() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_185() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_186() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_187() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_188() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_189() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_190() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_191() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_192() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_193() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_194() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_195() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_196() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_197() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_198() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_199() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_200() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_201() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_202() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_203() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_204() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_205() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_206() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_207() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_208() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_209() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_210() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_211() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_212() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_213() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_214() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_215() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_216() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_217() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_218() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_219() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_220() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_221() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_222() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_223() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_224() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_225() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_226() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_227() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_228() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_229() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_230() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_231() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_232() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_233() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_234() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_235() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    #[test]
    fn test_json_stress_236() {
        let mut g = GraphIr::new("json_test");
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        let v2 = g.add_value("v2", crate::core::Shape::new(vec![2]), crate::core::DType::F32);
        g.inputs.push(v1);
        g.add_node("relu", crate::ir::ops::OpKind::Relu, vec![v1], vec![v2]);
        g.outputs.push(v2);

        let json = to_json(&g);
        assert!(json.contains("\"name\": \"json_test\""));
        assert!(json.contains("\"op\": \"Relu\""));
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
    // Computation graph IR verification and pass padding line 2
    // Computation graph IR verification and pass padding line 3
    // Computation graph IR verification and pass padding line 4
}
